use std::{borrow::Cow, collections::VecDeque};

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

    #[serde(rename = "gpt-4.5-preview-2025-02-27")]
    Gpt4_5_preview_2025_02_27,

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
            | Self::o1Preview_2024_09_12
            | Self::Gpt4_5_preview_2025_02_27 => "developer",
        }
    }

    fn supports_temperature(&self) -> bool {
        match self {
            Self::Gpt4o
            | Self::Gpt4o_2024_08_06
            | Self::ChatGpt4oLatest
            | Self::Gpt4oMini_2024_07_18
            | Self::Gpt4oMini
            | Self::Gpt4_5_preview_2025_02_27 => true,
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
        chat: &[crate::Message],
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
        struct OpenAIToolCallFunction<'a> {
            name: &'a str,
            arguments: &'a str,
        }

        #[derive(Debug, serde::Serialize)]
        struct OpenAIToolCall<'a> {
            id: &'a str,
            r#type: &'a str,
            function: OpenAIToolCallFunction<'a>,
        }

        #[derive(Debug, serde::Serialize)]
        struct OpenAIMessage<'a> {
            role: &'a str,
            #[serde(skip_serializing_if = "str::is_empty")]
            content: Cow<'a, str>,
            #[serde(skip_serializing_if = "str::is_empty")]
            tool_call_id: &'a str,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            tool_calls: Vec<OpenAIToolCall<'a>>,
        }

        impl Default for OpenAIMessage<'_> {
            fn default() -> Self {
                Self {
                    role: "",
                    content: Cow::Borrowed(""),
                    tool_call_id: "",
                    tool_calls: vec![],
                }
            }
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

        let mut messages = vec![];

        if let Some(system_prompt) = system_prompt {
            messages.push(OpenAIMessage {
                role: &self.model.system_name(),
                content: Cow::Borrowed(system_prompt),
                ..OpenAIMessage::default()
            });
        }

        fn add_message<'a>(messages: &mut Vec<OpenAIMessage<'a>>, message: &'a crate::Message) {
            let new_message = match message {
                crate::Message::User(content) => {
                    // Try collate
                    if let Some(last) = messages.last_mut() {
                        if last.role == "user" {
                            if !last.content.is_empty() {
                                last.content =
                                    Cow::Owned(format!("{}\n\n{}", last.content, content));
                            } else {
                                last.content = Cow::Borrowed(content);
                            }

                            return;
                        }
                    }

                    OpenAIMessage {
                        role: "user",
                        content: Cow::Borrowed(content),
                        ..OpenAIMessage::default()
                    }
                }
                crate::Message::Assistant(content) => {
                    // Try collate
                    if let Some(last) = messages.last_mut() {
                        if last.role == "assistant" {
                            if !last.content.is_empty() {
                                last.content =
                                    Cow::Owned(format!("{}\n\n{}", last.content, content));
                            } else {
                                last.content = Cow::Borrowed(content);
                            }

                            return;
                        }
                    }

                    OpenAIMessage {
                        role: "assistant",
                        content: Cow::Borrowed(content),
                        ..OpenAIMessage::default()
                    }
                }
                crate::Message::ToolRequest {
                    id,
                    name,
                    arguments,
                } => {
                    let tool_request = OpenAIToolCall {
                        id: &id,
                        r#type: "function",
                        function: OpenAIToolCallFunction {
                            name,
                            arguments: &arguments.0,
                        },
                    };

                    // Try collate
                    if let Some(last) = messages.last_mut() {
                        if last.role == "assistant" {
                            last.tool_calls.push(tool_request);

                            return;
                        }
                    }

                    OpenAIMessage {
                        role: "assistant",
                        tool_calls: vec![tool_request],
                        ..OpenAIMessage::default()
                    }
                }
                crate::Message::ToolResponse { content, id } => OpenAIMessage {
                    role: "tool",
                    content: Cow::Borrowed(content),
                    tool_call_id: &id,
                    ..OpenAIMessage::default()
                },
            };

            messages.push(new_message);
        }

        for message in chat.iter() {
            add_message(&mut messages, message);
        }

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

        Ok(OpenAITokenStream::new(sse))
    }
}

pub struct OpenAITokenStream {
    stream: Option<std::pin::Pin<Box<SseClient>>>,
    outstanding: VecDeque<crate::Chunk>,
}

impl OpenAITokenStream {
    pub(crate) fn new(stream: SseClient) -> Self {
        Self {
            stream: Some(Box::pin(stream)),
            outstanding: VecDeque::new(),
        }
    }
}

impl futures::Stream for OpenAITokenStream {
    type Item = Result<crate::Chunk, crate::TokenError>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let Self {
            stream,
            outstanding,
        } = &mut *self;

        let Some(sse_client) = stream.as_mut() else {
            return std::task::Poll::Ready(None);
        };

        loop {
            // Return any outstanding chunks
            if let Some(chunk) = outstanding.pop_front() {
                return std::task::Poll::Ready(Some(Ok(chunk)));
            }

            let message = sse_client.as_mut().poll_next(cx);

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
                    let mut new_messages = match gather_messages(message.value.take()) {
                        Ok(new_messages) => new_messages,
                        Err(error) => {
                            self.stream = None;
                            return std::task::Poll::Ready(Some(Err(error)));
                        }
                    };

                    if new_messages.len() > 1 {
                        outstanding.extend(new_messages.drain(1..));
                    }
                    if let Some(message) = new_messages.into_iter().next() {
                        return std::task::Poll::Ready(Some(Ok(message)));
                    }

                    tracing::warn!(
                        "received empty message from endpoint: `{:?}`",
                        message.value
                    );
                }
                other => {
                    return std::task::Poll::Ready(Some(Err(crate::TokenError::UnknownEventType(
                        other.to_owned(),
                    ))))
                }
            }
        }
    }
}

fn gather_messages(mut value: serde_json::Value) -> Result<Vec<crate::Chunk>, crate::TokenError> {
    let Some(content) = value.as_object_mut() else {
        return Err(crate::TokenError::MalformedResponse {
            message: "expected OpenAI data to be an object",
            value,
        });
    };

    let Some(serde_json::Value::String(object)) = content.get("object") else {
        return Err(crate::TokenError::MalformedResponse {
            message: "expected OpenAI data to have object",
            value,
        });
    };

    match object.as_str() {
        "chat.completion.chunk" => {
            let Some(serde_json::Value::Array(choices)) = content.get_mut("choices") else {
                return Err(crate::TokenError::MalformedResponse {
                    message: "expected OpenAI chat completion chunk to have choices",
                    value,
                });
            };

            if choices.len() != 1 {
                return Err(crate::TokenError::MalformedResponse {
                    message: "expected OpenAI chat completion chunk to have exactly one choice",
                    value,
                });
            }

            let Some(serde_json::Value::Object(choice)) = choices.get_mut(0) else {
                return Err(crate::TokenError::MalformedResponse {
                    message: "expected OpenAI chat completion chunk to be an object",
                    value,
                });
            };

            let Some(serde_json::Value::Object(delta)) = choice.get_mut("delta") else {
                return Err(crate::TokenError::MalformedResponse {
                    message: "expected OpenAI chat completion chunk to have delta",
                    value,
                });
            };

            if let Some(serde_json::Value::String(text)) = delta.remove("content") {
                return Ok(if text.is_empty() {
                    vec![]
                } else {
                    vec![crate::Chunk::Token(text)]
                });
            };

            if let Some(serde_json::Value::Array(tool_calls)) = delta.get_mut("tool_calls") {
                return tool_calls
                    .into_iter()
                    .map(|tool_call| parse_tool_call(tool_call).map(crate::Chunk::ToolCall))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|message| crate::TokenError::MalformedResponse { message, value });
            };

            return Err(crate::TokenError::MalformedResponse {
                message: "expected OpenAI chat completion chunk delta to have known key",
                value,
            });
        }
        _ => {
            return Err(crate::TokenError::MalformedResponse {
                message: "unexpected OpenAI object",
                value,
            })
        }
    }
}

fn parse_tool_call(
    tool_call: &mut serde_json::Value,
) -> Result<crate::ToolCallChunk, &'static str> {
    let serde_json::Value::Object(tool_call) = tool_call else {
        return Err("expected tool call to be an object");
    };
    if let Some(serde_json::Value::String(ty)) = tool_call.get("type") {
        if ty != "function" {
            return Err("non-tool function calls are unsupported");
        }
    }

    let id = tool_call
        .get_mut("id")
        .and_then(JsonExt::take_str)
        .and_then(|v| (!v.is_empty()).then_some(v));

    let Some(serde_json::Value::Object(function)) = tool_call.get_mut("function") else {
        return Err("expected tool call to have object function");
    };

    let Some(arguments) = function.get_mut("arguments").and_then(JsonExt::take_str) else {
        return Err("expected tool call to have arguments");
    };

    let name = function
        .get_mut("name")
        .and_then(JsonExt::take_str)
        .and_then(|v| (!v.is_empty()).then_some(v));
    return Ok(crate::ToolCallChunk {
        id,
        name,
        arguments,
    });
}
