use hyper::{Method, Request, Version};

use crate::{sse::SseClient, JsonExt};

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

    fn supports_temperature(&self) -> bool {
        match self {
            Self::Gpt4o
            | Self::Gpt4o_2024_08_06
            | Self::ChatGpt4oLatest
            | Self::Gpt4oMini_2024_07_18
            | Self::Gpt4oMini => true,
            Self::o1
            | Self::o1_2024_12_17
            | Self::o1Mini
            | Self::o1Mini_2024_09_12
            | Self::o3Mini
            | Self::o3Mini_2025_01_31
            | Self::o1Preview
            | Self::o1Preview_2024_09_12 => false,
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
    type TokenStream = OpenAITokenStream;

    fn prompt(
        &self,
        chat: &[impl AsRef<str>],
        options: &crate::PromptOptions,
    ) -> Result<OpenAITokenStream, crate::PromptError> {
        let crate::PromptOptions {
            max_tokens,
            temperature,
            system_prompt,
            stopping_sequences,
            tools,
            reasoning,
        } = options;

        #[derive(Debug, serde::Serialize)]
        enum OpenAIReasoningEffort {
            #[serde(rename = "low")]
            Low,
            #[serde(rename = "medium")]
            Medium,
            #[serde(rename = "high")]
            High,
        }

        #[derive(Debug, serde::Serialize)]
        struct OpenAIFunctionDescription<'a> {
            name: &'a str,
            description: &'a str,
            parameters: &'a schemars::schema::Schema,
        }

        #[derive(Debug, serde::Serialize)]
        struct OpenAITool<'a> {
            r#type: &'a str,
            function: OpenAIFunctionDescription<'a>,
        }

        #[derive(Debug, serde::Serialize)]
        struct OpenAIMessage<'a> {
            role: &'a str,
            content: &'a str,
        }

        #[derive(Debug, serde::Serialize)]
        struct OpenAIRequest<'a> {
            model: GptModel,
            max_completion_tokens: usize,
            #[serde(skip_serializing_if = "Option::is_none")]
            temperature: Option<f32>,
            stream: bool,
            #[serde(skip_serializing_if = "<[String]>::is_empty")]
            stop: &'a [String],
            #[serde(skip_serializing_if = "Option::is_none")]
            reasoning_effort: Option<OpenAIReasoningEffort>,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            tools: Vec<OpenAITool<'a>>,
            messages: Vec<OpenAIMessage<'a>>,
        }

        let tools = tools
            .iter()
            .map(|tool| OpenAITool {
                r#type: "function",
                function: OpenAIFunctionDescription {
                    name: &tool.name,
                    description: &tool.description,
                    parameters: &tool.parameters.inner,
                },
            })
            .collect();

        let messages = system_prompt
            .iter()
            .map(|content| OpenAIMessage {
                role: &self.model.system_name(),
                content,
            })
            .chain(chat.iter().enumerate().map(|(i, content)| OpenAIMessage {
                role: if i % 2 == 0 { "user" } else { "assistant" },
                content: content.as_ref(),
            }))
            .collect();

        let body = OpenAIRequest {
            model: self.model,
            max_completion_tokens: *max_tokens,
            temperature: self.model.supports_temperature().then_some(*temperature),
            stop: stopping_sequences.as_slice(),
            stream: true,
            reasoning_effort: reasoning.map(|effort| match effort {
                crate::ReasoningEffort::Low => OpenAIReasoningEffort::Low,
                crate::ReasoningEffort::Medium => OpenAIReasoningEffort::Medium,
                crate::ReasoningEffort::High => OpenAIReasoningEffort::High,
            }),
            tools,
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

        Ok(OpenAITokenStream {
            stream: Some(Box::pin(sse)),
        })
    }
}

pub struct OpenAITokenStream {
    pub(super) stream: Option<std::pin::Pin<Box<SseClient>>>,
}

impl futures::Stream for OpenAITokenStream {
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
                "" => {
                    let content = message.value.as_object_mut().expect("message is an object");

                    let Some(serde_json::Value::String(object)) = content.get("object") else {
                        tracing::error!(
                            "expected OpenAI data to have object: {:#?}",
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
                                    "expected OpenAI chat completion chunk to have choices: {:#?}",
                                    message.value
                                );
                                continue;
                            };

                            if choices.len() != 1 {
                                tracing::error!(
                                    "expected OpenAI chat completion chunk to have exactly one choice: {:#?}",
                                    message.value
                                );
                                continue;
                            }

                            let Some(serde_json::Value::Object(choice)) = choices.get_mut(0) else {
                                tracing::error!(
                                    "expected OpenAI chat completion chunk to have at least one choice: {:#?}",
                                    message.value
                                );
                                continue;
                            };

                            let Some(serde_json::Value::Object(delta)) = choice.get_mut("delta")
                            else {
                                tracing::error!(
                                    "expected OpenAI chat completion chunk to have delta: {:#?}",
                                    message.value
                                );
                                continue;
                            };

                            if let Some(serde_json::Value::String(text)) = delta.remove("content") {
                                if text.is_empty() {
                                    continue;
                                }

                                return std::task::Poll::Ready(Some(Ok(crate::Chunk::Token(text))));
                            };

                            if let Some(serde_json::Value::Array(mut tool_calls)) =
                                delta.remove("tool_calls")
                            {
                                if tool_calls.len() != 1 {
                                    unimplemented!(
                                        "expected OpenAI chat completion chunk to have exactly one tool call: {:#?}",
                                        message.value
                                    );
                                }
                                let serde_json::Value::Object(mut tool_call) = tool_calls[0].take()
                                else {
                                    tracing::error!(
                                        "expected tool call to be an object: {:#?}",
                                        message.value
                                    );
                                    continue;
                                };
                                if let Some(serde_json::Value::String(ty)) =
                                    tool_call.remove("type")
                                {
                                    if ty != "function" {
                                        unimplemented!("non-tool function calls are unsupported");
                                    }
                                }

                                let Some(serde_json::Value::Object(mut function)) =
                                    tool_call.remove("function")
                                else {
                                    tracing::error!(
                                        "expected tool call to have object function: {:#?}",
                                        message.value
                                    );
                                    continue;
                                };

                                let id = tool_call
                                    .remove("id")
                                    .and_then(|mut v| v.take_str())
                                    .and_then(|v| (!v.is_empty()).then_some(v));

                                let name = function
                                    .remove("name")
                                    .and_then(|mut v| v.take_str())
                                    .and_then(|v| (!v.is_empty()).then_some(v));
                                let Some(arguments) =
                                    function.remove("arguments").and_then(|mut v| v.take_str())
                                else {
                                    tracing::error!(
                                        "expected tool call to have arguments: {:#?}",
                                        message.value
                                    );
                                    continue;
                                };
                                return std::task::Poll::Ready(Some(Ok(crate::Chunk::ToolCall(
                                    crate::ToolCallChunk {
                                        id,
                                        name,
                                        arguments,
                                    },
                                ))));
                            };

                            tracing::error!(
                                    "expected OpenAI chat completion chunk delta to have known key: {:#?}",
                                    message.value
                                );
                        }
                        other => tracing::error!(
                            "unexpected OpenAI object: `{other}` with value {:#?}",
                            content
                        ),
                    }
                }
                other => tracing::error!(
                    "unexpected OpenAI event: `{other}` with value {:#?}",
                    message.value
                ),
            }
        }
    }
}
