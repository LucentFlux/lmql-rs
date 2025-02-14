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

pub struct PromptOptions {
    pub max_tokens: usize,
    pub temperature: f32,
    pub system_prompt: Option<String>,
    pub stopping_sequences: Vec<String>,
}

impl Default for PromptOptions {
    fn default() -> Self {
        Self {
            max_tokens: DEFAULT_MAX_TOKENS,
            temperature: DEFAULT_TEMPERATURE,
            system_prompt: None,
            stopping_sequences: vec![],
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
    pub fn add_stopping_sequence(&mut self, stopping_sequence: String) -> &mut Self {
        self.stopping_sequences.push(stopping_sequence);
        self
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }
    pub fn temperature(&self) -> f32 {
        self.temperature
    }
    pub fn system_prompt(&self) -> Option<&String> {
        self.system_prompt.as_ref()
    }
    pub fn stopping_sequences(&self) -> &[String] {
        &self.stopping_sequences
    }
}

/// Some hook into an LLM, which can be used to generate text.
pub trait LLM {
    type TokenStream<'a>: futures::Stream<Item = Result<Token, TokenError>> + Send + 'a;

    /// Generates a response to the given prompt. The prompt is a list of strings, where each
    /// is either the user or the assistant, starting with the user and alternating.
    fn prompt<'a>(
        &self,
        prompt: &'a [&'a str],
        options: PromptOptions,
    ) -> Result<Self::TokenStream<'a>, PromptError>;
}

mod sealed {
    pub trait TokenStreamExtSealed {}
    impl<T> TokenStreamExtSealed for T where
        T: futures::Stream<Item = Result<super::Token, super::TokenError>> + Send
    {
    }
}
/// Utility methods for token stream sources.
pub trait TokenStreamExt: sealed::TokenStreamExtSealed {
    /// Converts the stream of tokens into a single string future. This is useful for
    /// when you don't want to filter the tokens as they arrive.
    fn all_tokens(self) -> impl std::future::Future<Output = Result<String, TokenError>> + Send;
}
impl<T> TokenStreamExt for T
where
    T: sealed::TokenStreamExtSealed + futures::Stream<Item = Result<Token, TokenError>> + Send,
{
    async fn all_tokens(self) -> Result<String, TokenError> {
        use futures::StreamExt;
        self.fold(Ok(String::new()), |acc, token| async {
            let mut acc = acc?;
            acc.push_str(&token?.0);
            Ok(acc)
        })
        .await
    }
}

#[derive(Debug)]
pub struct Token(pub String);

#[derive(Debug, thiserror::Error)]
pub enum TokenError {
    #[error("the connection was lost")]
    ConnectionLost(#[from] sse::Error),
}
pub use sse::Error as SseError;
