use std::path::Path;

use minijinja::{context, Environment, Error, ErrorKind, Value as JinjaValue};
use serde_json::Value;
use tokenizers::Tokenizer;

/// ChatML, which Qwen models use, for checkpoints that ship no template.
const CHATML: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n\
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    fn to_value(&self) -> JinjaValue {
        JinjaValue::from_pairs([
            ("role", self.role.as_str()),
            ("content", self.content.as_str()),
        ])
    }
}

/// A model's Jinja chat template, rendered the way `transformers` renders it.
pub struct ChatTemplate {
    env: Environment<'static>,
}

impl ChatTemplate {
    pub fn new(source: String) -> Result<Self, Error> {
        let mut env = Environment::new();
        // Hugging Face templates call Python string methods and raise_exception.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function(
            "raise_exception",
            |message: String| -> Result<String, Error> {
                Err(Error::new(ErrorKind::InvalidOperation, message))
            },
        );
        env.add_template_owned("chat", source)?;
        Ok(Self { env })
    }

    pub fn chatml() -> Self {
        Self::new(CHATML.to_string()).expect("the built-in template is valid")
    }

    /// The template from `tokenizer_config.json` or `chat_template.jinja`, if any.
    pub fn from_model_dir(dir: &Path) -> Result<Option<Self>, Error> {
        if let Ok(source) = std::fs::read_to_string(dir.join("chat_template.jinja")) {
            return Self::new(source).map(Some);
        }
        let source = read_json(&dir.join("tokenizer_config.json"))
            .and_then(|config| config.get("chat_template")?.as_str().map(str::to_string));
        source.map(Self::new).transpose()
    }

    pub fn render(&self, messages: &[ChatMessage]) -> Result<String, Error> {
        self.env.get_template("chat")?.render(context! {
            messages => messages.iter().map(ChatMessage::to_value).collect::<Vec<_>>(),
            add_generation_prompt => true,
        })
    }
}

fn read_json(path: &Path) -> Option<Value> {
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}

/// Every token id the checkpoint treats as end of sequence: `eos_token_id`
/// from `generation_config.json` and `config.json`, plus Qwen's chat markers.
pub fn stop_tokens(dir: &Path, tokenizer: &Tokenizer) -> Vec<u32> {
    let mut ids = Vec::new();
    for file in ["generation_config.json", "config.json"] {
        match read_json(&dir.join(file)).and_then(|c| c.get("eos_token_id").cloned()) {
            Some(Value::Number(n)) => ids.extend(n.as_u64().map(|n| n as u32)),
            Some(Value::Array(list)) => {
                ids.extend(list.iter().filter_map(|v| v.as_u64()).map(|n| n as u32))
            }
            _ => {}
        }
    }
    ids.extend(
        ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|t| tokenizer.token_to_id(t)),
    );
    ids.sort_unstable();
    ids.dedup();
    ids
}
