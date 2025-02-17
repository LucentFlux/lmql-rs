use std::fmt::Display;

use hyper::{Method, Request, Version};

use crate::sse::SseClient;

pub struct OpenRouter {
    model: String,
    bearer_header: String,
}

impl OpenRouter {
    /// Sugar for [`Self::new`], but uses the `OPENROUTER_API_KEY` environment variable for the API key.
    pub fn new_from_env(model: impl Into<String>) -> Self {
        Self::new(
            model,
            std::env::var("OPENROUTER_API_KEY")
                .expect("OPENROUTER_API_KEY environment variable not set"),
        )
    }

    pub fn new(model: impl Into<String>, api_key: impl Display) -> Self {
        Self {
            model: model.into(),
            bearer_header: format!("Bearer {api_key}"),
        }
    }
}

impl crate::LLM for OpenRouter {
    type TokenStream<'a> = OpenRouterTokenStream;

    fn prompt<'a>(
        &self,
        prompt: &'a [&'a str],
        options: crate::PromptOptions,
    ) -> Result<OpenRouterTokenStream, crate::PromptError> {
        #[derive(Debug, serde::Serialize)]
        struct OpenRouterMessage<'a> {
            role: &'a str,
            content: &'a str,
        }

        #[derive(Debug, serde::Serialize)]
        struct OpenRouterRequest<'a> {
            model: &'a str,
            max_tokens: usize,
            temperature: f32,
            stream: bool,
            #[serde(skip_serializing_if = "<[&'a str]>::is_empty")]
            stop: &'a [&'a str],
            messages: Vec<OpenRouterMessage<'a>>,
            include_reasoning: bool,
        }

        let messages = options
            .system_prompt
            .iter()
            .map(|content| OpenRouterMessage {
                role: "system",
                content,
            })
            .chain(
                prompt
                    .iter()
                    .enumerate()
                    .map(|(i, &content)| OpenRouterMessage {
                        role: if i % 2 == 0 { "user" } else { "assistant" },
                        content,
                    }),
            )
            .collect();

        let body = OpenRouterRequest {
            model: &self.model,
            max_tokens: options.max_tokens,
            temperature: options.temperature,
            stop: options.stopping_sequences,
            stream: true,
            include_reasoning: false,
            messages,
        };
        let body = serde_json::to_string(&body)?;
        tracing::debug!("OpenRouter request body: {}", body);

        let request = Request::builder()
            .uri("https://openrouter.ai/api/v1/chat/completions")
            .header("Authorization", &self.bearer_header)
            .header("content-type", "application/json")
            .version(Version::HTTP_2)
            .method(Method::POST)
            .body(body)?;
        tracing::debug!("OpenRouter request: {:#?}", request);
        let sse = SseClient::spawn(request);

        Ok(OpenRouterTokenStream {
            stream: Some(Box::pin(sse)),
        })
    }
}

pub struct OpenRouterTokenStream {
    stream: Option<std::pin::Pin<Box<SseClient>>>,
}

impl futures::Stream for OpenRouterTokenStream {
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
                "" => {
                    let content = message.value.as_object_mut().expect("message is an object");

                    let Some(serde_json::Value::String(object)) = content.get("object") else {
                        tracing::error!(
                            "expected OpenRouter data to have object: {:#?}",
                            message.value
                        );
                        continue;
                    };

                    match object.as_str() {
                        "chat.completion.chunk" => {
                            let Some(serde_json::Value::Array(choices)) =
                                content.get_mut("choices")
                            else {
                                tracing::error!(
                                    "expected OpenRouter chat completion chunk to have choices: {:#?}",
                                    message.value
                                );
                                continue;
                            };

                            if choices.len() != 1 {
                                tracing::error!(
                                    "expected OpenRouter chat completion chunk to have exactly one choice: {:#?}",
                                    message.value
                                );
                                continue;
                            }

                            let Some(serde_json::Value::Object(choice)) = choices.get_mut(0) else {
                                tracing::error!(
                                    "expected OpenRouter chat completion chunk to have at least one choice: {:#?}",
                                    message.value
                                );
                                continue;
                            };

                            let Some(serde_json::Value::Object(delta)) = choice.get_mut("delta")
                            else {
                                tracing::error!(
                                    "expected OpenRouter chat completion chunk to have delta: {:#?}",
                                    message.value
                                );
                                continue;
                            };

                            let Some(serde_json::Value::String(text)) = delta.remove("content")
                            else {
                                tracing::error!(
                                    "expected OpenRouter chat completion chunk to have content: {:#?}",
                                    message.value
                                );
                                continue;
                            };

                            return std::task::Poll::Ready(Some(Ok(crate::Token(text))));
                        }
                        other => tracing::error!(
                            "unexpected OpenRouter object: `{other}` with value {:#?}",
                            content
                        ),
                    }
                }
                other => tracing::error!(
                    "unexpected OpenRouter event: `{other}` with value {:#?}",
                    message.value
                ),
            }
        }
    }
}
