use std::{borrow::Cow, fmt::Display};

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
    type TokenStream = super::openai::OpenAITokenStream;

    fn prompt(
        &self,
        chat: &[crate::Message],
        options: &crate::PromptOptions,
    ) -> Result<super::openai::OpenAITokenStream, crate::PromptError> {
        let crate::PromptOptions {
            max_tokens,
            temperature,
            system_prompt,
            stopping_sequences,
            tools,
            reasoning,
        } = options;

        #[derive(Debug, serde::Serialize)]
        enum OpenRouterReasoningEffort {
            #[serde(rename = "low")]
            Low,
            #[serde(rename = "medium")]
            Medium,
            #[serde(rename = "high")]
            High,
        }

        #[derive(Debug, serde::Serialize)]
        struct OpenRouterFunctionDescription<'a> {
            name: &'a str,
            description: &'a str,
            parameters: &'a schemars::schema::Schema,
        }

        #[derive(Debug, serde::Serialize)]
        struct OpenRouterTool<'a> {
            r#type: &'a str,
            function: OpenRouterFunctionDescription<'a>,
        }

        #[derive(Debug, serde::Serialize)]
        struct OpenRouterReasoning {
            effort: OpenRouterReasoningEffort,
        }

        #[derive(Debug, serde::Serialize)]
        struct OpenRouterToolCallFunction<'a> {
            name: &'a str,
            arguments: &'a str,
        }

        #[derive(Debug, serde::Serialize)]
        struct OpenRouterToolCall<'a> {
            id: &'a str,
            r#type: &'a str,
            function: OpenRouterToolCallFunction<'a>,
        }

        #[derive(Debug, serde::Serialize)]
        struct OpenRouterMessage<'a> {
            role: &'a str,
            #[serde(skip_serializing_if = "str::is_empty")]
            content: Cow<'a, str>,
            #[serde(skip_serializing_if = "str::is_empty")]
            tool_call_id: &'a str,
            #[serde(skip_serializing_if = "Vec::is_empty")]
            tool_calls: Vec<OpenRouterToolCall<'a>>,
        }

        impl Default for OpenRouterMessage<'_> {
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
        struct OpenRouterRequest<'a> {
            model: &'a str,
            max_tokens: usize,
            temperature: f32,
            stream: bool,
            #[serde(skip_serializing_if = "<[String]>::is_empty")]
            stop: &'a [String],
            tools: Vec<OpenRouterTool<'a>>,
            reasoning: Option<OpenRouterReasoning>,
            messages: Vec<OpenRouterMessage<'a>>,
        }

        let tools = tools
            .iter()
            .map(|tool| OpenRouterTool {
                r#type: "function",
                function: OpenRouterFunctionDescription {
                    name: &tool.name,
                    description: &tool.description,
                    parameters: &tool.parameters.inner,
                },
            })
            .collect();

        let mut messages = vec![];
        if let Some(system_prompt) = system_prompt {
            messages.push(OpenRouterMessage {
                role: "system",
                content: Cow::Borrowed(system_prompt),
                ..OpenRouterMessage::default()
            });
        }

        fn try_append_text<'a>(
            messages: &mut Vec<OpenRouterMessage<'a>>,
            content: &'a str,
            role: &'a str,
        ) -> Option<OpenRouterMessage<'a>> {
            if content.is_empty() {
                return None;
            }

            // Try collate
            if let Some(last) = messages.last_mut() {
                if last.role == role {
                    if !last.content.is_empty() {
                        last.content = Cow::Owned(format!("{}\n\n{}", last.content, content));
                    } else {
                        last.content = Cow::Borrowed(content);
                    }
                    return None;
                }
            }

            Some(OpenRouterMessage {
                role,
                content: Cow::Borrowed(content),
                ..OpenRouterMessage::default()
            })
        }

        fn add_message<'a>(messages: &mut Vec<OpenRouterMessage<'a>>, message: &'a crate::Message) {
            let new_message = match message {
                crate::Message::User(content) => {
                    let Some(message) = try_append_text(messages, content, "user") else {
                        return;
                    };
                    message
                }
                crate::Message::Assistant(content) => {
                    let Some(message) = try_append_text(messages, content, "assistant") else {
                        return;
                    };
                    message
                }
                crate::Message::ToolRequest {
                    id,
                    name,
                    arguments,
                } => {
                    let tool_request = OpenRouterToolCall {
                        id: &id,
                        r#type: "function",
                        function: OpenRouterToolCallFunction {
                            name,
                            arguments: &arguments.serialized,
                        },
                    };

                    // Try collate
                    if let Some(last) = messages.last_mut() {
                        if last.role == "assistant" {
                            last.tool_calls.push(tool_request);

                            return;
                        }
                    }

                    OpenRouterMessage {
                        role: "assistant",
                        tool_calls: vec![tool_request],
                        ..OpenRouterMessage::default()
                    }
                }
                crate::Message::ToolResponse { content, id } => OpenRouterMessage {
                    role: "tool",
                    content: Cow::Borrowed(content),
                    tool_call_id: &id,
                    ..OpenRouterMessage::default()
                },
            };

            messages.push(new_message);
        }

        for message in chat.iter() {
            add_message(&mut messages, message);
        }

        let body = OpenRouterRequest {
            model: &self.model,
            max_tokens: *max_tokens,
            temperature: *temperature,
            stop: stopping_sequences.as_slice(),
            stream: true,
            tools,
            reasoning: reasoning.map(|effort| OpenRouterReasoning {
                effort: match effort {
                    crate::ReasoningEffort::Low => OpenRouterReasoningEffort::Low,
                    crate::ReasoningEffort::Medium => OpenRouterReasoningEffort::Medium,
                    crate::ReasoningEffort::High => OpenRouterReasoningEffort::High,
                },
            }),
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

        Ok(super::openai::OpenAITokenStream::new(sse))
    }
}
