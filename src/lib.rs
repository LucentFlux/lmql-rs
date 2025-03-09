#![doc = include_str!("../README.md")]

pub mod llms;
mod sse;

pub const DEFAULT_MAX_TOKENS: usize = 4096;
pub const DEFAULT_TEMPERATURE: f32 = 1.0;

//pub use lmql_macros::*;
//#[macro_export]
/*macro_rules! prompt {
    ($model:expr => $(
        user: $prompt:literal;
        assistant: $response:literal $(where $($out:ident : $out_ty:ty),* $(,)?)?
    );* $(;)?) => {async {
        let res = $crate::prompt_inner!($model => $(
            user: $prompt;
            assistant: $response $(where $($out : $out_ty),*)*;
        )*).await;

        // Formatting in IDE.
        if let Ok(res) = res {
            if false {
                $(
                    let _ = format!($prompt);
                    let _ = format!($response, $($($out = res.$out),*)* );
                )*
            }
        }

        res
    }};
}*/

#[derive(Debug, thiserror::Error)]
pub enum PromptError {
    #[error("failed to build request to model")]
    RequestError(#[from] hyper::http::Error),
    #[error("failed to transcode prompt or response")]
    TranscodingError(#[from] serde_json::Error),
}

pub struct ToolParameter<'a> {
    pub name: &'a str,
    pub description: &'a str,
    pub parameters: &'a serde_json::Value,
}

/// The parameters of a tool available to an LLM.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolParameters {
    inner: schemars::schema::Schema,
}

impl ToolParameters {
    pub fn new<S: schemars::JsonSchema>() -> Self {
        let mut generator = schemars::gen::SchemaGenerator::default();
        Self {
            inner: <S as schemars::JsonSchema>::json_schema(&mut generator),
        }
    }
}

/// A tool accessible to an LLM.
#[derive(Debug, Clone, PartialEq)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: ToolParameters,
}

/// The effort to put into reasoning.
/// For non-reasoning models, this is ignored.
/// For non-open-ai models, this corresponds to the maximum number of tokens to use for reasoning.
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    fn max_tokens(&self) -> usize {
        match self {
            Self::Low => 1024,
            Self::Medium => 2048,
            Self::High => 4096,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PromptOptions {
    pub max_tokens: usize,
    pub temperature: f32,
    pub system_prompt: Option<String>,
    pub stopping_sequences: Vec<String>,
    pub tools: Vec<Tool>,
    pub reasoning: Option<ReasoningEffort>,
}

impl Default for PromptOptions {
    fn default() -> Self {
        Self {
            max_tokens: DEFAULT_MAX_TOKENS,
            temperature: DEFAULT_TEMPERATURE,
            system_prompt: None,
            stopping_sequences: vec![],
            tools: vec![],
            reasoning: None,
        }
    }
}

impl PromptOptions {
    pub fn set_max_tokens(&mut self, max_tokens: usize) -> &mut Self {
        self.max_tokens = max_tokens;
        self
    }
    pub fn set_temperature(&mut self, temperature: f32) -> &mut Self {
        self.temperature = temperature;
        self
    }
    pub fn set_system_prompt(&mut self, system_prompt: String) -> &mut Self {
        self.system_prompt = Some(system_prompt);
        self
    }
    pub fn set_stopping_sequences(&mut self, stopping_sequences: Vec<String>) -> &mut Self {
        self.stopping_sequences = stopping_sequences;
        self
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }
    pub fn temperature(&self) -> f32 {
        self.temperature
    }
    pub fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }
    pub fn stopping_sequences(&self) -> &[String] {
        &self.stopping_sequences[..]
    }
}

/// Some `serde_json::Value` that has been serialized to a string.
pub struct SerializedJson(String);

impl SerializedJson {
    /// Serialization can fail if T's implementation of Serialize decides to fail, or if T contains a map with non-string keys.
    pub fn try_new(value: serde_json::Value) -> serde_json::Result<Self> {
        Ok(Self(serde_json::to_string(&value)?))
    }
}

pub enum Message {
    User(String),
    Assistant(String),
    ToolRequest {
        id: String,
        name: String,
        arguments: SerializedJson,
    },
    ToolResponse {
        content: String,
        id: String,
    },
}

/// Some hook into an LLM, which can be used to generate text.
pub trait LLM {
    type TokenStream: futures::Stream<Item = Result<Chunk, TokenError>> + Send;

    /// Generates a response to the given prompt. The prompt is a list of strings, where each
    /// is either the user or the assistant, starting with the user and alternating.
    fn prompt(
        &self,
        messages: &[Message],
        options: &PromptOptions,
    ) -> Result<Self::TokenStream, PromptError>;
}

mod sealed {
    pub trait TokenStreamExtSealed {}
    impl<T> TokenStreamExtSealed for T where
        T: futures::Stream<Item = Result<super::Chunk, super::TokenError>> + Send
    {
    }
}
/// Utility methods for token stream sources.
pub trait TokenStreamExt: sealed::TokenStreamExtSealed {
    /// Converts the stream of tokens into a single set of tokens future, collapsing adjacent like tokens.
    /// This is useful for when you don't want to filter the tokens as they arrive.
    fn all_tokens(self)
        -> impl std::future::Future<Output = Result<Vec<Chunk>, TokenError>> + Send;
}
impl<T> TokenStreamExt for T
where
    T: sealed::TokenStreamExtSealed + futures::Stream<Item = Result<Chunk, TokenError>> + Send,
{
    async fn all_tokens(self) -> Result<Vec<Chunk>, TokenError> {
        use futures::StreamExt;
        let mut stream = Box::pin(self);

        let mut acc = vec![];

        while let Some(token) = stream.next().await {
            tracing::debug!("received token in all_tokens: {:?}", token);
            if let Some(last_acc) = acc.last_mut() {
                match (last_acc, token?) {
                    (Chunk::Token(lhs), Chunk::Token(rhs)) => lhs.push_str(&rhs),
                    (Chunk::Thinking(lhs), Chunk::Thinking(rhs)) => lhs.push_str(&rhs),
                    (Chunk::ToolCall(lhs), Chunk::ToolCall(rhs))
                        if lhs.id.as_ref().is_none_or(|lhs_id| {
                            rhs.id.as_ref().is_none_or(|rhs_id| lhs_id == rhs_id)
                        }) =>
                    {
                        lhs.id = lhs.id.take().or(rhs.id);
                        lhs.name = lhs.name.take().or(rhs.name);
                        lhs.arguments.push_str(&rhs.arguments);
                    }
                    (_, token) => acc.push(token),
                }
            } else {
                acc.push(token?);
            };
        }

        Ok(acc)
    }
}

#[derive(Debug, Clone)]
pub struct ToolCallChunk {
    pub id: Option<String>,
    pub name: Option<String>,
    pub arguments: String,
}

#[derive(Debug, Clone)]
pub enum Chunk {
    Token(String),
    Thinking(String),
    ToolCall(ToolCallChunk),
}

impl Chunk {
    pub fn try_into_message(self) -> Option<Message> {
        match self {
            Chunk::Token(content) => Some(Message::Assistant(content)),
            Chunk::Thinking(_) => None,
            Chunk::ToolCall(tool_call_chunk) => Some(Message::ToolRequest {
                id: tool_call_chunk.id?,
                name: tool_call_chunk.name?,
                arguments: SerializedJson::try_new(
                    serde_json::from_str::<serde_json::Value>(&tool_call_chunk.arguments).ok()?,
                )
                .ok()?,
            }),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TokenError {
    #[error("the connection was lost")]
    ConnectionLost(#[from] sse::Error),
    #[error("the server responded with an unknown event type `{0}`")]
    UnknownEventType(String),
    #[error("the server responded with unexpected data: {message}")]
    MalformedResponse {
        message: &'static str,
        value: serde_json::Value,
    },
}

pub use schemars::JsonSchema;
pub use serde;
pub use serde_json;
pub use sse::Error as SseError;

trait JsonExt {
    fn take_str(&mut self) -> Option<String>;
}

impl JsonExt for serde_json::Value {
    fn take_str(&mut self) -> Option<String> {
        if let serde_json::Value::String(s) = self.take() {
            Some(s)
        } else {
            None
        }
    }
}
