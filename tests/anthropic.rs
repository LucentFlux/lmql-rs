mod common;

mod sonnet37 {
    crate::tests_with_llm! {
        lmql::llms::anthropic::Claude::new_from_env(
            lmql::llms::anthropic::ClaudeModel::Claude_3_7_Sonnet_20250219,
        )
    }
}

mod haiku35 {
    crate::tests_with_llm! {
        lmql::llms::anthropic::Claude::new_from_env(
            lmql::llms::anthropic::ClaudeModel::Claude_3_5_Haiku_20241022,
        )

        => skip reasoning
    }
}
