use futures::StreamExt;
use lmql::{PromptOptions, Token, LLM};

#[tokio::main]
async fn main() {
    tracing_subscriber::FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO)
        .init();

    let gpt = lmql::llms::openai::Gpt::new_from_env(lmql::llms::openai::GptModel::Gpt4o);
    let mut stream = gpt
        .prompt(
            &["What is the molecular formula for hemoglobin?"],
            PromptOptions::default(),
        )
        .unwrap();

    while let Some(t) = stream.next().await {
        match t {
            Ok(Token(t)) => print!("{}", t),
            Err(e) => panic!("{e:#?}"),
        }
    }
}
