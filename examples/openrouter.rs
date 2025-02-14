use futures::StreamExt;
use lmql::{PromptOptions, Token, LLM};

#[tokio::main]
async fn main() {
    tracing_subscriber::FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO)
        .init();

    let gpt =
        lmql::llms::openrouter::OpenRouter::new_from_env("nousresearch/hermes-3-llama-3.1-405b");
    let mut stream = gpt
        .prompt(
            &["What kind of fruit fell on Newton's head?"],
            PromptOptions::default(),
        )
        .unwrap();

    while let Some(t) = stream.next().await {
        match t {
            Ok(Token(t)) => print!("{}", t),
            Err(e) => panic!("{e:#?}"),
        }
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
}
