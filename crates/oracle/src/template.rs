//! Render a model's own Hugging Face chat template, with upstream's arguments — so the prompt the
//! oracle reads is byte-for-byte the prompt upstream's Python renders, not our reading of it.

/// Render one user turn with the generation prompt, as upstream's `Codec.render` does.
pub fn render_user(template: &str, user: &str, kwargs: &serde_json::Value) -> Result<String, String> {
    let mut env = minijinja::Environment::new();
    // Hugging Face templates call Python string methods (`.split`, `.rstrip`, `.startswith`) and
    // `raise_exception`; both are supplied here as transformers supplies them.
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function("raise_exception", |msg: String| -> Result<String, minijinja::Error> {
        Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
    });
    env.add_template("chat", template).map_err(|e| e.to_string())?;
    let mut ctx = serde_json::json!({
        "messages": [{"role": "user", "content": user}],
        "add_generation_prompt": true,
    });
    if let (Some(c), Some(k)) = (ctx.as_object_mut(), kwargs.as_object()) {
        c.extend(k.clone());
    }
    env.get_template("chat")
        .and_then(|t| t.render(minijinja::value::Serde(&ctx)))
        .map_err(|e| format!("{e:#}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/latentport-4-27b/tokenizers/");

    fn template(side: &str) -> Option<String> {
        std::fs::read_to_string(format!("{DIR}{side}/chat_template.jinja")).ok()
    }

    #[test]
    fn a_minimal_chatml_template_renders_one_user_turn() {
        let t = "{%- for m in messages %}{{- '<|im_start|>' + m.role + '\\n' + m.content + '<|im_end|>\\n' }}{%- endfor %}\
                 {%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}";
        let out = render_user(t, "hi", &serde_json::json!({})).unwrap();
        assert_eq!(out, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    #[ignore = "integration"]
    fn upstreams_templates_render_with_upstreams_arguments() {
        let (Some(src), Some(tgt)) = (template("source"), template("target")) else {
            panic!("no upstream templates under {DIR} — run `just oracle-models`")
        };
        let s = render_user(&src, "hi", &serde_json::json!({})).unwrap();
        assert_eq!(s, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n");
        let t = render_user(&tgt, "hi", &serde_json::json!({"preserve_thinking": false, "reasoning_effort": "medium"})).unwrap();
        assert!(t.ends_with("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"), "{t:?}");
        assert!(render_user(&tgt, "hi", &serde_json::json!({"reasoning_effort": "extreme"})).is_err());
    }
}
