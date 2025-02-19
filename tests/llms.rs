use futures::StreamExt;
use lmql::{PromptOptions, TokenStreamExt, LLM};

#[tokio::test]
async fn anthropic_stream() {
    let claude = lmql::llms::anthropic::Claude::new_from_env(
        lmql::llms::anthropic::ClaudeModel::Claude_3_5_Haiku_20241022,
    );
    let stream = claude
        .prompt(&["Hello!"], PromptOptions::default())
        .unwrap();
    let response = stream.all_tokens().await.unwrap();
    println!("{:?}", response);
}

#[tokio::test]
async fn openrouter_stream() {
    let r1 = lmql::llms::openrouter::OpenRouter::new_from_env("deepseek/deepseek-r1");
    let mut stream = r1.prompt(&["Hello!"], PromptOptions::default()).unwrap();
    while let Some(token) = stream.next().await {
        print!("{:?}", token.unwrap());
    }
}
