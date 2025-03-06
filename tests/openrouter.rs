mod common;

mod llama {
    super::tests_with_llm! {
        lmql::llms::openrouter::OpenRouter::new_from_env("meta-llama/llama-3.2-3b-instruct")
            => skip tool, reasoning
    }
}

mod qwen {
    super::tests_with_llm! {
        lmql::llms::openrouter::OpenRouter::new_from_env("qwen/qwen-turbo")
    }
}
