# lmql-rs
An LLM programming language inspired by the Python library of the same name

## Features

- [x] Multiple backend support, including Anthropic and OpenRouter
- [x] Stream-based async API to allow for real-time response
- [x] Stream cancelling to avoid wasting tokens on a bad response
- [ ] Macros for a prompt DSL like the LMQL Python library

## Usage

```Rust
use futures::StreamExt;
use lmql::{PromptOptions, Token, LLM};

#[tokio::main]
async fn main() {
    tracing_subscriber::FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO)
        .init();

    let claude = lmql::llms::anthropic::Claude::new_from_env(
        lmql::llms::anthropic::ClaudeModel::Claude_3_5_Haiku_20241022,
    );
    let mut stream = claude
        .prompt(
            &["Please provide a poem about the moon."],
            PromptOptions::default(),
        )
        .unwrap();

    while let Some(t) = stream.next().await {
        match t {
            Ok(Token(t)) => print!("{}", t),
            Err(e) => {
                println!("Error from stream: {e:#?}");
                break;
            },
        }
    }
}
```