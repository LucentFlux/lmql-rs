use proc_macro::TokenStream;

// Used inside of `prompt` in the outer crate
#[proc_macro]
#[doc(hidden)]
pub fn prompt_inner(args: TokenStream) -> TokenStream {
    args
}
