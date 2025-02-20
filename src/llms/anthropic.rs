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
    type TokenStream<'a> = ClaudeTokenStream;

    fn prompt<'a>(
        &self,
        chat: &'a [impl AsRef<str> + 'a],
        options: crate::PromptOptions,
    ) -> Result<ClaudeTokenStream, crate::PromptError> {
        #[derive(Debug, serde::Serialize)]
        struct ClaudeMessage<'a> {
            role: &'a str,
            content: &'a str,
        }

        fn is_one(v: &f32) -> bool {
            *v == 1.0
        }

        #[derive(Debug, serde::Serialize)]
        struct ClaudeRequest<'a> {
            model: ClaudeModel,
            max_tokens: usize,
            #[serde(skip_serializing_if = "is_one")]
            temperature: f32,
            #[serde(skip_serializing_if = "std::ops::Not::not")]
            stream: bool,
            #[serde(skip_serializing_if = "<[&'a str]>::is_empty")]
            stop_sequences: &'a [&'a str],
            #[serde(skip_serializing_if = "Option::is_none")]
            system_prompt: Option<&'a str>,
            messages: Vec<ClaudeMessage<'a>>,
        }

        let body = ClaudeRequest {
            model: self.model,
            max_tokens: options.max_tokens,
            temperature: options.temperature,
            stop_sequences: options.stopping_sequences,
            system_prompt: options.system_prompt,
            stream: true,
            messages: chat
                .iter()
                .enumerate()
                .map(|(i, content)| ClaudeMessage {
                    role: if i % 2 == 0 { "user" } else { "assistant" },
                    content: content.as_ref(),
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

        Ok(ClaudeTokenStream {
            stream: Some(Box::pin(sse)),
        })
    }
}

pub struct ClaudeTokenStream {
    stream: Option<std::pin::Pin<Box<SseClient>>>,
}

impl futures::Stream for ClaudeTokenStream {
    type Item = Result<crate::Token, crate::TokenError>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        loop {
            let Some(stream) = self.stream.as_mut() else {
                return std::task::Poll::Ready(None);
            };

            let message = stream.as_mut().poll_next(cx);

            let message = match message {
                std::task::Poll::Ready(None) => {
                    self.stream = None;
                    return std::task::Poll::Ready(None);
                }
                std::task::Poll::Ready(Some(message)) => message,
                std::task::Poll::Pending => return std::task::Poll::Pending,
            };

            let mut message = match message {
                Err(error) => {
                    self.stream = None;
                    return std::task::Poll::Ready(Some(Err(crate::TokenError::ConnectionLost(
                        error,
                    ))));
                }
                Ok(message) => message,
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
                    return std::task::Poll::Ready(Some(Ok(crate::Token(text))));
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
                    return std::task::Poll::Ready(Some(Ok(crate::Token(text))));
                }
                "content_block_stop" | "message_delta" => { /* pass */ }
                "message_stop" => {
                    self.stream = None;
                    return std::task::Poll::Ready(None);
                }
                other => tracing::error!(
                    "unexpected anthropic event: `{other}` with value {:#?}",
                    message.value
                ),
            }
        }
    }
}
