use futures::StreamExt;
use lmql::{PromptOptions, LLM};

#[tokio::test]
async fn anthropic_stream() {
    let claude = lmql::llms::anthropic::Claude::new_from_env(
        lmql::llms::anthropic::ClaudeModel::Claude_3_5_Haiku_20241022,
    );
    let mut stream = claude
        .prompt(&["Hello!"], PromptOptions::default())
        .unwrap();
    while let Some(token) = stream.next().await {
        println!("{:?}", token.unwrap());
    }
}

#[tokio::test]
async fn openai_stream() {
    let gpt =
        lmql::llms::openai::Gpt::new_from_env(lmql::llms::openai::GptModel::Gpt4oMini_2024_07_18);
    let mut stream = gpt.prompt(&["Hello!"], PromptOptions::default()).unwrap();
    while let Some(token) = stream.next().await {
        println!("{:?}", token.unwrap());
    }
}
