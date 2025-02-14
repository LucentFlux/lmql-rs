use futures::StreamExt;
use lmql::{PromptOptions, Token, LLM};

#[tokio::main]
async fn main() {
    tracing_subscriber::FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO)
        .init();

    let hermes =
        lmql::llms::openrouter::OpenRouter::new_from_env("nousresearch/hermes-3-llama-3.1-405b");
    let mut stream = hermes
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

    let deepseek = lmql::llms::openrouter::OpenRouter::new_from_env("deepseek/deepseek-r1");
    let mut stream = deepseek
        .prompt(
            &["What is the molecular formula for hemoglobin?"],
            PromptOptions::default(),
        )
        .unwrap();

    while let Some(t) = stream.next().await {
        match t {
            Ok(Token(t)) => {
                if !t.is_empty() {
                    print!("{}", t);
                    let _ = std::io::Write::flush(&mut std::io::stdout());
                }
            }
            Err(e) => panic!("{e:#?}"),
        }
    }
}
