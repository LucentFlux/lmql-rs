use lmql::{PromptOptions, TokenStream, LLM};

#[tokio::test]
async fn anthropic_stream() {
    let claude = lmql::llms::anthropic::Claude::new_from_env(
        lmql::llms::anthropic::ClaudeModel::Claude_3_5_Haiku_20241022,
    );
    let mut stream = claude
        .prompt(&["Hello!"], PromptOptions::default())
        .unwrap();
    while let Some(token) = stream.next_token().await {
        println!("{:?}", token.unwrap());
    }
}
