use lmql::{PromptOptions, TokenStreamExt};

#[macro_export]
macro_rules! tests_with_llm {
    ($llm:expr $(=> skip $($test_name:ident),* $(,)?)?) => {
        use crate::common::*;

        $($(
            async fn $test_name(_llm: impl lmql::LLM) {}
        )*)*

        #[tokio::test]
        async fn test_stream() {
            setup();
            stream($llm).await;
        }
        #[tokio::test]
        async fn test_reasoning() {
            setup();
            reasoning($llm).await;
        }
        #[tokio::test]
        async fn test_tool() {
            setup();
            tool($llm).await;
        }
    };
}

pub fn setup() {
    let _ = tracing_subscriber::FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO)
        .with_test_writer()
        .try_init();
}

pub async fn stream(llm: impl lmql::LLM) {
    let stream = llm
        .prompt(
            &[lmql::Message::User("Hello!".into())],
            &PromptOptions::default(),
        )
        .unwrap();
    let response = stream.all_tokens().await.unwrap();
    assert_eq!(response.len(), 1, "{response:?}");
    assert!(matches!(&response[0], lmql::Chunk::Token(text) if text.len() > 1));
}

pub async fn reasoning(llm: impl lmql::LLM) {
    let stream = llm
        .prompt(
            &[lmql::Message::User(
                "How many atoms of iron are in a molecule of hemoglobin?".into(),
            )],
            &PromptOptions {
                reasoning: Some(lmql::ReasoningEffort::Low),
                temperature: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
    let mut response = stream.all_tokens().await.unwrap();

    assert!(response.len() >= 1 && response.len() <= 2, "{response:?}");

    if response.len() > 1 {
        let reasoning = response.remove(0);
        assert!(matches!(reasoning, lmql::Chunk::Thinking(_)));
    }

    let lmql::Chunk::Token(text) = &response[0] else {
        panic!("Expected a text response, got {response:?}");
    };
    assert!(text.contains("4") || text.contains("four"), "`{text}`");
    assert!(!text.contains("<thinking>"), "`{text}`");
}

pub async fn tool(llm: impl lmql::LLM) {
    #[derive(lmql::JsonSchema, serde::Deserialize)]
    struct StockPrice {
        /// The stock ticker symbol, e.g. AAPL for Apple Inc.
        ticker: String,
    }
    let description = "Retrieves the current stock price for a given ticker symbol.
    The ticker symbol must be a valid symbol for a publicly traded company on a major US stock exchange like NYSE or NASDAQ.
    The tool will return the latest trade price in USD.
    It should be used when the user asks about the current or most recent price of a specific stock.
    It will not provide any other information about the stock or company.".to_string();
    let stream = llm
        .prompt(
            &[lmql::Message::User(
                "What is the current price of AAPL?".into(),
            )],
            &PromptOptions {
                tools: vec![lmql::Tool {
                    name: "get_stock_price".to_string(),
                    description,
                    parameters: lmql::ToolParameters::new::<StockPrice>(),
                }],
                max_tokens: 4000,
                temperature: 0.12,
                system_prompt: Some("You are an assistant with little to no knowledge about the world. but you are very good at using tools to answer questions.".to_owned()),
                stopping_sequences: vec![
                    "pear".to_owned(),
                    "banana".to_owned(),
                ],
                reasoning: None
            },
        )
        .unwrap();
    let mut response = stream.all_tokens().await.unwrap();
    assert!(response.len() <= 2, "{response:?}");

    if response.len() > 1 {
        let text = response.remove(0);
        assert!(matches!(text, lmql::Chunk::Token(_)));
    }

    let lmql::Chunk::ToolCall(lmql::ToolCallChunk {
        id: _,
        name,
        arguments,
    }) = &response[0]
    else {
        panic!("Expected a tool call, got {response:?}");
    };

    assert_eq!(name.as_deref(), Some("get_stock_price"));

    let arguments = serde_json::from_str::<StockPrice>(arguments).unwrap();
    assert_eq!(arguments.ticker, "AAPL");
}
