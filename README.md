# lmql-rs
An typesafe high-level LLM API for Rust, inspired by the Python library of the same name.

## Features

- [x] Multiple backend support, including Anthropic, OpenAI and OpenRouter
- [x] Async and Stream support, with cancelling to avoid wasting tokens on a bad response
- [x] Tools, with a type-safe interface
- [ ] Macros for a prompt DSL like the LMQL Python library

## Usage

```rust
use futures::StreamExt;
use lmql::{PromptOptions, Chunk, Message, LLM};

#[tokio::main]
async fn main() {
    let claude = lmql::llms::anthropic::Claude::new_from_env(
        lmql::llms::anthropic::ClaudeModel::Claude_3_5_Haiku_20241022,
    );
    let mut stream = claude
        .prompt(
            &[Message::User("Please provide a poem about the moon.".into())],
            &PromptOptions::default(),
        )
        .unwrap();

    // Loop over each token as they arrive
    while let Some(t) = stream.next().await {
        if let Ok(Chunk::Token(t)) = t {
            print!("{}", t)
        } else {
            panic!("Unexpected chunk: {t:#?}")
        }
    }

    // Or use `lmql::TokenStreamExt` to collect the tokens together
    let mut stream = claude
        .prompt(
            &[Message::User("What is bitcoin?".into())],
            &PromptOptions::default(),
        )
        .unwrap();

    use lmql::TokenStreamExt;
    let response = stream.all_tokens().await.unwrap();
    assert_eq!(response.len(), 1);
    let Chunk::Token(t) = &response[0] else {
        panic!("Expected only text in response")
    };
    println!("{t}");
}
```