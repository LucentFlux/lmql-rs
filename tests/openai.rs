mod common;

mod gpt4o {
    super::tests_with_llm! {
        lmql::llms::openai::Gpt::new_from_env(
            lmql::llms::openai::GptModel::Gpt4oMini,
        )

        => skip reasoning
    }
}

mod o3 {
    super::tests_with_llm! {
        lmql::llms::openai::Gpt::new_from_env(
            lmql::llms::openai::GptModel::o3Mini,
        )
    }
}
