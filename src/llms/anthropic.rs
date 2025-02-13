use hyper::{Method, Request, Version};

use crate::sse::SseClient;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ClaudeModel {
    #[serde(rename = "claude-3-5-sonnet-20241022")]
    Claude_3_5_Sonnet_20241022,
    #[serde(rename = "claude-3-5-sonnet-latest")]
    Claude_3_5_Sonnet_Latest,

    #[serde(rename = "claude-3-5-haiku-20241022")]
    Claude_3_5_Haiku_20241022,
    #[serde(rename = "claude-3-5-haiku-latest")]
    Claude_3_5_Haiku_Latest,

    #[serde(rename = "claude-3-opus-20240229")]
    Claude_3_Opus_20240229,
    #[serde(rename = "claude-3-opus-latest")]
    Claude_3_Opus_Latest,

    #[serde(rename = "claude-3-sonnet-20240229")]
    Claude_3_Sonnet_20240229,

    #[serde(rename = "claude-3-haiku-20240307")]
    Claude_3_Haiku_20240307,
}

pub struct Claude {
    model: ClaudeModel,
    api_key: String,
}

impl Claude {
    /// Sugar for [`Self::new`], but uses the `ANTHROPIC_API_KEY` environment variable for the API key.
    pub fn new_from_env(model: ClaudeModel) -> Self {
        Self::new(
            model,
            std::env::var("ANTHROPIC_API_KEY")
                .expect("ANTHROPIC_API_KEY environment variable not set"),
        )
    }

    pub fn new(model: ClaudeModel, api_key: String) -> Self {
        Self { model, api_key }
    }
}

impl crate::LLM for Claude {
    fn prompt<'a>(
        &self,
        prompt: &'a [&'a str],
        options: crate::PromptOptions,
    ) -> Result<impl crate::TokenStream + 'a, crate::PromptError> {
        #[derive(Debug, serde::Serialize)]
        struct ClaudeMessage {
            role: String,
            content: String,
        }

        fn is_one(v: &f32) -> bool {
            *v == 1.0
        }

        #[derive(Debug, serde::Serialize)]
        struct ClaudeRequest {
            model: ClaudeModel,
            max_tokens: usize,
            #[serde(skip_serializing_if = "is_one")]
            temperature: f32,
            #[serde(skip_serializing_if = "std::ops::Not::not")]
            stream: bool,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            stop_sequences: Vec<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            system_prompt: Option<String>,
            messages: Vec<ClaudeMessage>,
        }

        let body = ClaudeRequest {
            model: self.model,
            max_tokens: options.max_tokens,
            temperature: options.temperature,
            stop_sequences: options.stopping_sequences,
            system_prompt: options.system_prompt,
            stream: true,
            messages: prompt
                .iter()
                .enumerate()
                .map(|(i, &content)| ClaudeMessage {
                    role: if i % 2 == 0 { "user" } else { "assistant" }.to_string(),
                    content: content.to_string(),
                })
                .collect(),
        };
        let body = serde_json::to_string(&body)?;
        tracing::debug!("Claude request body: {}", body);

        let request = Request::builder()
            .uri("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .version(Version::HTTP_2)
            .method(Method::POST)
            .body(body)?;
        tracing::debug!("Claude request: {:#?}", request);
        let sse = SseClient::spawn(request);

        Ok(ClaudeTokenStream { stream: Some(sse) })
    }
}

struct ClaudeTokenStream {
    stream: Option<SseClient>,
}

impl crate::TokenStream for ClaudeTokenStream {
    fn next_token<'a>(
        &'a mut self,
    ) -> impl std::future::Future<Output = Option<Result<crate::Token, crate::TokenError>>> + Send + 'a
    {
        async {
            loop {
                let Some(stream) = self.stream.as_mut() else {
                    return None;
                };

                let message = stream.next().await;

                let mut message = match message {
                    crate::sse::Event::Failed(error) => {
                        self.stream = None;
                        return Some(Err(crate::TokenError::ConnectionLost(error)));
                    }
                    crate::sse::Event::Shutdown => {
                        self.stream = None;
                        return None;
                    }
                    crate::sse::Event::Message(message) => message,
                };

                match message.event.as_str() {
                    "ping" => {}
                    "message_start" => { /* pass */ }
                    "content_block_start" => {
                        let content = message
                            .value
                            .as_object_mut()
                            .expect("content block start is an object")
                            .get_mut("content_block")
                            .expect("content block start has content block")
                            .as_object_mut()
                            .expect("content block is an object");
                        assert_eq!(
                            content.get("type"),
                            Some(&serde_json::Value::String("text".to_string()))
                        );
                        let serde_json::Value::String(text) = content
                            .get_mut("text")
                            .expect("content block has text")
                            .take()
                        else {
                            panic!("content block text is a string");
                        };
                        return Some(Ok(crate::Token(text)));
                    }
                    "content_block_delta" => {
                        let content = message
                            .value
                            .as_object_mut()
                            .expect("content block delta is an object")
                            .get_mut("delta")
                            .expect("content block delta has delta")
                            .as_object_mut()
                            .expect("delta is an object");
                        assert_eq!(
                            content.get("type"),
                            Some(&serde_json::Value::String("text_delta".to_string()))
                        );
                        let serde_json::Value::String(text) =
                            content.get_mut("text").expect("delta has text").take()
                        else {
                            panic!("delta text is a string");
                        };
                        return Some(Ok(crate::Token(text)));
                    }
                    "content_block_stop" | "message_delta" | "message_stop" => return None,
                    other => tracing::error!(
                        "unexpected anthropic event: `{other}` with value {:#?}",
                        message.value
                    ),
                }
            }
        }
    }
}
