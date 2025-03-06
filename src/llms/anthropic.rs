use hyper::{Method, Request, Version};

use crate::{sse::SseClient, JsonExt};

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ClaudeModel {
    #[serde(rename = "claude-3-7-sonnet-20250219")]
    Claude_3_7_Sonnet_20250219,
    #[serde(rename = "claude-3-7-sonnet-latest")]
    Claude_3_7_Sonnet_latest,

    #[serde(rename = "claude-3-5-sonnet-20241022")]
    Claude_3_5_Sonnet_20241022,
    #[serde(rename = "claude-3-5-sonnet-20240620")]
    Claude_3_5_Sonnet_20240620,
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
    type TokenStream = ClaudeTokenStream;

    fn prompt(
        &self,
        chat: &[impl AsRef<str>],
        options: &crate::PromptOptions,
    ) -> Result<ClaudeTokenStream, crate::PromptError> {
        let crate::PromptOptions {
            max_tokens,
            temperature,
            system_prompt,
            stopping_sequences,
            tools,
            reasoning,
        } = options;

        fn is_one(v: &f32) -> bool {
            *v == 1.0
        }

        #[derive(Debug, serde::Serialize)]
        struct ClaudeThinking {
            r#type: &'static str,
            budget_tokens: usize,
        }

        #[derive(Debug, serde::Serialize)]
        struct ClaudeTool<'a> {
            name: &'a str,
            description: &'a str,
            input_schema: &'a schemars::schema::Schema,
        }

        #[derive(Debug, serde::Serialize)]
        struct ClaudeMessage<'a> {
            role: &'a str,
            content: &'a str,
        }

        #[derive(Debug, serde::Serialize)]
        struct ClaudeRequest<'a> {
            model: ClaudeModel,
            max_tokens: usize,
            #[serde(skip_serializing_if = "is_one")]
            temperature: f32,
            #[serde(skip_serializing_if = "std::ops::Not::not")]
            stream: bool,
            #[serde(skip_serializing_if = "<[String]>::is_empty")]
            stop_sequences: &'a [String],
            #[serde(skip_serializing_if = "Option::is_none")]
            system: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            thinking: Option<ClaudeThinking>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            tools: Vec<ClaudeTool<'a>>,
            messages: Vec<ClaudeMessage<'a>>,
        }

        let tools = tools
            .iter()
            .map(|tool| ClaudeTool {
                name: &tool.name,
                description: &tool.description,
                input_schema: &tool.parameters.inner,
            })
            .collect();

        let body = ClaudeRequest {
            model: self.model,
            max_tokens: *max_tokens,
            temperature: if reasoning.is_none() {
                *temperature
            } else {
                1.0
            },
            stop_sequences: stopping_sequences.as_slice(),
            system: system_prompt.as_deref(),
            stream: true,
            thinking: reasoning.map(|level| ClaudeThinking {
                r#type: "enabled",
                budget_tokens: level.max_tokens(),
            }),
            tools,
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
    type Item = Result<crate::Chunk, crate::TokenError>;

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
                    let Some(content) = message.value.as_object_mut() else {
                        tracing::error!("content block start should be an object - {message:?}");
                        continue;
                    };
                    let Some(content) = content.get_mut("content_block") else {
                        tracing::error!(
                            "content block start should have content_block - {content:?}"
                        );
                        continue;
                    };
                    let Some(content) = content.as_object_mut() else {
                        tracing::error!("content block should be an object - {content:?}");
                        continue;
                    };

                    let Some(token) = process_content_block(content) else {
                        continue;
                    };

                    return std::task::Poll::Ready(Some(Ok(token)));
                }
                "content_block_delta" => {
                    let Some(content) = message.value.as_object_mut() else {
                        tracing::error!("content block delta should be an object - {message:?}");
                        continue;
                    };
                    let Some(content) = content.get_mut("delta") else {
                        tracing::error!("content block delta should have delta - {content:?}");
                        continue;
                    };
                    let Some(content) = content.as_object_mut() else {
                        tracing::error!("delta should be an object - {content:?}");
                        continue;
                    };

                    let Some(token) = process_content_block(content) else {
                        continue;
                    };

                    return std::task::Poll::Ready(Some(Ok(token)));
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

fn process_content_block(
    content: &mut serde_json::Map<String, serde_json::Value>,
) -> Option<crate::Chunk> {
    let Some(&serde_json::Value::String(ref ty)) = content.get("type") else {
        tracing::error!("expected content block to have type - {content:?}");
        return None;
    };

    match ty.as_str() {
        "text" | "text_delta" => {
            let Some(text) = content.get_mut("text").and_then(|text| text.take_str()) else {
                tracing::error!("expected content text block to have text - {content:?}");
                return None;
            };

            if text.is_empty() {
                return None;
            }

            Some(crate::Chunk::Token(text))
        }
        "thinking" | "thinking_delta" => {
            let Some(thinking) = content.get_mut("thinking").and_then(|text| text.take_str())
            else {
                tracing::error!("expected content thinking block to have thinking - {content:?}");
                return None;
            };

            if thinking.is_empty() {
                return None;
            }

            Some(crate::Chunk::Thinking(thinking))
        }
        "tool_use" => {
            let id = content.get_mut("id").and_then(|id| id.take_str());
            let name = content.get_mut("name").and_then(|id| id.take_str());

            // Check we weren't given an input block
            if let Some(serde_json::Value::Object(input)) = content.get("input") {
                if !input.is_empty() {
                    tracing::error!("expected content tool_use input to be empty - {content:?}");
                }
            } else {
                tracing::error!(
                    "expected content tool_use block to have empty input - {content:?}"
                );
            };

            Some(crate::Chunk::ToolCall(crate::ToolCallChunk {
                id,
                name,
                arguments: String::new(),
            }))
        }
        "input_json_delta" => {
            let Some(ty) = content.get_mut("type").and_then(|ty| ty.take_str()) else {
                tracing::error!("expected json_delta to have a type - {content:?}");
                return None;
            };
            if ty != "input_json_delta" {
                tracing::error!("expected json_delta to have type input_json_delta - {content:?}");
                return None;
            }
            let Some(json) = content.get_mut("partial_json").and_then(|id| id.take_str()) else {
                tracing::error!(
                    "expected content input_json_delta block to have partial_json - {content:?}"
                );
                return None;
            };

            Some(crate::Chunk::ToolCall(crate::ToolCallChunk {
                id: None,
                name: None,
                arguments: json,
            }))
        }
        "signature_delta" | "redacted_thinking" => None,
        _ => {
            tracing::error!("unknown content block type: {ty} - {content:?}");
            None
        }
    }
}
