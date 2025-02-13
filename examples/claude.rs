use lmql::{PromptOptions, Token, TokenStream, LLM};

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

    while let Some(t) = stream.next_token().await {
        match t {
            Ok(Token(t)) => print!("{}", t),
            Err(e) => panic!("{e:#?}"),
        }
    }
}
