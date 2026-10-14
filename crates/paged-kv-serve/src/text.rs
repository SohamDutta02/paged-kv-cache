use std::sync::Arc;

use tokenizers::Tokenizer;

/// Turns tokens into text as they arrive. Decoding is redone over the whole
/// output each time, since a token's text can depend on its neighbours, and
/// output ending in a partial UTF-8 character is held back.
pub struct Detokenizer {
    tokenizer: Arc<Tokenizer>,
    tokens: Vec<u32>,
    emitted: String,
}

impl Detokenizer {
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            emitted: String::new(),
        }
    }

    /// The text added by `token`, possibly empty.
    pub fn push(&mut self, token: u32) -> String {
        self.tokens.push(token);
        let Ok(text) = self.tokenizer.decode(&self.tokens, true) else {
            return String::new();
        };
        if text.ends_with('\u{FFFD}') || !text.starts_with(&self.emitted) {
            return String::new();
        }
        let delta = text[self.emitted.len()..].to_string();
        self.emitted = text;
        delta
    }
}
