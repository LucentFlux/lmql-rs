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
    type TokenStream = super::openai::OpenAITokenStream;

    fn prompt(
        &self,
        chat: &[impl AsRef<str>],
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

        let messages = system_prompt
            .iter()
            .map(|content| OpenRouterMessage {
                role: "system",
                content,
            })
            .chain(
                chat.iter()
                    .enumerate()
                    .map(|(i, content)| OpenRouterMessage {
                        role: if i % 2 == 0 { "user" } else { "assistant" },
                        content: content.as_ref(),
                    }),
            )
            .collect();

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

        Ok(super::openai::OpenAITokenStream {
            stream: Some(Box::pin(sse)),
        })
    }
}
