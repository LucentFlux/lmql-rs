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
        => skip tool
    }
}

mod gemini {
    super::tests_with_llm! {
        lmql::llms::openrouter::OpenRouter::new_from_env("google/gemini-2.0-flash-lite-001")
    }
}
