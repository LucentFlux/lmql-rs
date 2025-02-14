use hyper::{Method, Request, Version};

use crate::sse::SseClient;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum GptModel {
    #[serde(rename = "gpt-4o-2024-08-06")]
    Gpt4o_2024_08_06,
    #[serde(rename = "gpt-4o")]
    Gpt4o,
    #[serde(rename = "chatgpt-4o-latest")]
    ChatGpt4oLatest,
    #[serde(rename = "gpt-4o-mini-2024-07-18")]
    Gpt4oMini_2024_07_18,
    #[serde(rename = "gpt-4o-mini")]
    Gpt4oMini,

    #[serde(rename = "o1-2024-12-17")]
    o1_2024_12_17,
    #[serde(rename = "o1")]
    o1,

    #[serde(rename = "o1-mini-2024-09-12")]
    o1Mini_2024_09_12,
    #[serde(rename = "o1-mini")]
    o1Mini,

    #[serde(rename = "o3-mini-2025-01-31")]
    o3Mini_2025_01_31,
    #[serde(rename = "o3-mini")]
    o3Mini,

    #[serde(rename = "o1-preview-2024-09-12")]
    o1Preview_2024_09_12,
    #[serde(rename = "o1-preview")]
    o1Preview,
}

impl GptModel {
    fn system_name(&self) -> &'static str {
        match self {
            Self::Gpt4o
            | Self::Gpt4o_2024_08_06
            | Self::ChatGpt4oLatest
            | Self::Gpt4oMini_2024_07_18
            | Self::Gpt4oMini => "system",
            Self::o1
            | Self::o1_2024_12_17
            | Self::o1Mini
            | Self::o1Mini_2024_09_12
            | Self::o3Mini
            | Self::o3Mini_2025_01_31
            | Self::o1Preview
            | Self::o1Preview_2024_09_12 => "developer",
        }
    }
}

pub struct Gpt {
    model: GptModel,
    bearer_header: String,
}

impl Gpt {
    /// Sugar for [`Self::new`], but uses the `OPENAI_API_KEY` environment variable for the API key.
    pub fn new_from_env(model: GptModel) -> Self {
        Self::new(
            model,
            std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set"),
        )
    }

    pub fn new(model: GptModel, api_key: String) -> Self {
        Self {
            model,
            bearer_header: format!("Bearer {api_key}"),
        }
    }
}

impl crate::LLM for Gpt {
    type TokenStream<'a> = GptTokenStream;

    fn prompt<'a>(
        &self,
        prompt: &'a [&'a str],
        options: crate::PromptOptions,
    ) -> Result<GptTokenStream, crate::PromptError> {
        #[derive(Debug, serde::Serialize)]
        struct GptMessage<'a> {
            role: &'a str,
            content: &'a str,
        }

        fn is_one(v: &f32) -> bool {
            *v == 1.0
        }

        #[derive(Debug, serde::Serialize)]
        struct GptRequest<'a> {
            model: GptModel,
            max_tokens: usize,
            #[serde(skip_serializing_if = "is_one")]
            temperature: f32,
            #[serde(skip_serializing_if = "std::ops::Not::not")]
            stream: bool,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            stop_sequences: Vec<String>,
            messages: Vec<GptMessage<'a>>,
        }

        let messages = options
            .system_prompt
            .iter()
            .map(|content| GptMessage {
                role: self.model.system_name(),
                content,
            })
            .chain(prompt.iter().enumerate().map(|(i, &content)| GptMessage {
                role: if i % 2 == 0 { "user" } else { "assistant" },
                content,
            }))
            .collect();

        let body = GptRequest {
            model: self.model,
            max_tokens: options.max_tokens,
            temperature: options.temperature,
            stop_sequences: options.stopping_sequences,
            stream: true,
            messages,
        };
        let body = serde_json::to_string(&body)?;
        tracing::debug!("OpenAI request body: {}", body);

        let request = Request::builder()
            .uri("https://api.openai.com/v1/chat/completions")
            .header("Authorization", &self.bearer_header)
            .header("content-type", "application/json")
            .version(Version::HTTP_2)
            .method(Method::POST)
            .body(body)?;
        tracing::debug!("OpenAI request: {:#?}", request);
        let sse = SseClient::spawn(request);

        Ok(GptTokenStream {
            stream: Some(Box::pin(sse)),
        })
    }
}

pub struct GptTokenStream {
    stream: Option<std::pin::Pin<Box<SseClient>>>,
}

impl futures::Stream for GptTokenStream {
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
