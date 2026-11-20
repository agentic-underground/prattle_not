//! Turning a person's words into a model prompt, and a growing decode back into new text.

/// SmolLM2-Instruct's chat template, for one user turn, ending where the assistant speaks.
pub fn chat_prompt(user: &str) -> String {
    format!(
        "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n\
         <|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    )
}

/// A model family's prompt for one user turn, ending where the assistant speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Template {
    /// SmolLM2-Instruct's (`model_type` "llama" — the only Llama-family models here).
    SmolLM2,
    /// Qwen3's, with thinking off: the assistant turn opens on an empty think block.
    Qwen3,
    /// Llama 3's header-delimited turn (a `llama` config declaring `rope_type: llama3`).
    Llama3,
    /// Gemma 3's turn (`gemma3_text`, or a multimodal `gemma3` whose text_config is).
    Gemma3,
    /// Gemma 4's plain turn — E2B's template with thinking off. The 31B's opens an empty thought channel
    /// as well; on the plain turn it writes that channel itself and answers, while E2B given the channel
    /// reasons in the open instead of answering. One form for the family: a bridged pair reads one prompt.
    Gemma4,
    /// Granite 4.0's role-delimited turn, opening on its chat template's default system message.
    Granite,
    /// LFM2's: ChatML after its `<|startoftext|>`.
    Lfm2,
}

impl Template {
    /// The family a model's `config.json` declares. An unknown family is an error, never a default.
    pub fn from_config(config_json: &[u8]) -> Result<Template, String> {
        let v: serde_json::Value = serde_json::from_slice(config_json).map_err(|e| e.to_string())?;
        let kind = v["text_config"]["model_type"].as_str().or(v["model_type"].as_str());
        if kind.is_some_and(|k| k.starts_with("gemma3")) {
            return Ok(Template::Gemma3);
        }
        if kind.is_some_and(|k| k.starts_with("gemma4")) {
            return Ok(Template::Gemma4);
        }
        match v["model_type"].as_str() {
            Some("llama") if v["rope_scaling"]["rope_type"] == "llama3" => Ok(Template::Llama3),
            Some("llama") => Ok(Template::SmolLM2),
            Some("qwen3") | Some("qwen3_5") => Ok(Template::Qwen3),
            Some("granitemoehybrid") => Ok(Template::Granite),
            Some("lfm2") => Ok(Template::Lfm2),
            Some(other) => Err(format!("no prompt template for model_type {other:?}")),
            None => Err("config.json has no model_type".into()),
        }
    }

    /// As `prompt`, opening on `system` as the system turn — each family's own way of writing one
    /// (Gemma has no system role: its template puts the system text at the head of the user turn).
    pub fn prompt_with_system(&self, system: &str, user: &str) -> String {
        match self {
            Template::SmolLM2 => format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"),
            Template::Qwen3 => format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            Template::Gemma3 => format!("<bos><start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n<start_of_turn>model\n"),
            Template::Gemma4 => format!("<bos><|turn>system\n{system}<turn|>\n<|turn>user\n{user}<turn|>\n<|turn>model\n"),
            Template::Lfm2 => format!("<|startoftext|><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"),
            Template::Granite => format!(
                "<|start_of_role|>system<|end_of_role|>{system}<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>{user}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
            ),
            Template::Llama3 => format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
        }
    }

    /// Everything the prompt holds before the user's own words: the template's head and, with `system`,
    /// the system turn.
    pub fn head(&self, system: Option<&str>) -> String {
        // split at a marker, so the head is the template's own and never a second copy of it
        const USER: &str = "\u{1}USER\u{1}";
        let whole = match system {
            Some(system) => self.prompt_with_system(system, USER),
            None => self.prompt(USER),
        };
        whole.split(USER).next().unwrap_or_default().to_string()
    }

    /// What closes an assistant turn: the family's end-of-turn token — generation stops on it without
    /// reading it — and whatever its template writes after it.
    pub fn end_reply(&self) -> &'static str {
        match self {
            Template::SmolLM2 | Template::Qwen3 | Template::Lfm2 => "<|im_end|>\n",
            Template::Gemma3 => "<end_of_turn>\n",
            Template::Gemma4 => "<turn|>\n",
            Template::Granite => "<|end_of_text|>\n",
            Template::Llama3 => "<|eot_id|>",
        }
    }

    /// A user turn as it follows an earlier exchange, ending where the assistant speaks: `prompt`
    /// without its head (no `<bos>`, no system turn).
    pub fn user_turn(&self, user: &str) -> String {
        let opens = match self {
            Template::SmolLM2 | Template::Qwen3 | Template::Lfm2 => "<|im_start|>user",
            Template::Gemma3 => "<start_of_turn>user",
            Template::Gemma4 => "<|turn>user",
            Template::Granite => "<|start_of_role|>user",
            Template::Llama3 => "<|start_header_id|>user",
        };
        let whole = self.prompt(user);
        // the head ends in this turn's opener; anything before it opens the conversation once
        let head = self.head(None);
        let at = whole[..head.len()].rfind(opens).unwrap_or(head.len());
        whole[at..].to_string()
    }

    pub fn prompt(&self, user: &str) -> String {
        match self {
            Template::SmolLM2 => chat_prompt(user),
            Template::Qwen3 => format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            Template::Gemma3 => format!("<bos><start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"),
            Template::Gemma4 => format!("<bos><|turn>user\n{user}<turn|>\n<|turn>model\n"),
            Template::Lfm2 => format!("<|startoftext|><|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"),
            Template::Granite => format!(
                "<|start_of_role|>system<|end_of_role|>You are a helpful assistant. Please ensure responses are professional, accurate, and safe.<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>{user}<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
            ),
            Template::Llama3 => format!(
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
        }
    }
}

/// The text `now` adds to `prev`. A decode of more tokens can rewrite its tail (a multi-byte
/// character completing), so this yields nothing until the new decode extends the old one.
pub fn delta<'a>(prev: &str, now: &'a str) -> &'a str {
    now.strip_prefix(prev).unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prompt_carries_the_question_and_ends_where_the_assistant_speaks() {
        let p = chat_prompt("What is 2+2?");
        assert!(p.contains("<|im_start|>user\nWhat is 2+2?<|im_end|>"));
        assert!(p.ends_with("<|im_start|>assistant\n"));
    }

    const FAMILIES: [Template; 7] = [Template::SmolLM2, Template::Qwen3, Template::Llama3, Template::Gemma3, Template::Gemma4, Template::Granite, Template::Lfm2];

    /// The head is the prompt up to the user's first word, with and without a system turn, in every family.
    #[test]
    fn the_head_is_the_prompt_up_to_the_users_words() {
        for t in FAMILIES {
            let (with, without) = (t.prompt_with_system("Answer in French.", "Hi there"), t.prompt("Hi there"));
            let (hw, ho) = (t.head(Some("Answer in French.")), t.head(None));
            assert_eq!(with.strip_prefix(&hw).map(|r| r.starts_with("Hi there")), Some(true), "{t:?}: {hw:?} is not the head of {with:?}");
            assert_eq!(without.strip_prefix(&ho).map(|r| r.starts_with("Hi there")), Some(true), "{t:?}: {ho:?} is not the head of {without:?}");
            assert!(hw.contains("in French"), "{t:?}: the head lost the system turn");
        }
    }

    /// A second exchange is written as the family's own template writes a conversation: the reply closed
    /// by its end-of-turn token, the next user turn with no second head.
    #[test]
    fn a_second_turn_follows_the_first_as_the_familys_template_writes_it() {
        let two = |t: Template, system: Option<&str>| {
            let first = match system {
                Some(s) => t.prompt_with_system(s, "Hi"),
                None => t.prompt("Hi"),
            };
            format!("{first}Hello!{}{}", t.end_reply(), t.user_turn("Bye"))
        };
        assert_eq!(
            two(Template::Qwen3, None),
            "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHello!<|im_end|>\n<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert_eq!(
            two(Template::Gemma3, Some("Be brief.")),
            "<bos><start_of_turn>user\nBe brief.\n\nHi<end_of_turn>\n<start_of_turn>model\nHello!<end_of_turn>\n<start_of_turn>user\nBye<end_of_turn>\n<start_of_turn>model\n"
        );
        assert_eq!(
            two(Template::Llama3, None),
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHello!<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nBye<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        assert_eq!(
            two(Template::Granite, None).matches("<|start_of_role|>system").count(),
            1,
            "Granite's default system turn opens the conversation once"
        );
    }

    #[test]
    fn a_qwen3_config_gets_qwen3s_turn_with_thinking_off() {
        let t = Template::from_config(br#"{"model_type": "qwen3"}"#).unwrap();
        assert_eq!(t.prompt("Hi"), "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }

    #[test]
    fn a_llama_config_keeps_smollm2s_prompt_and_an_unknown_family_refuses() {
        assert_eq!(Template::from_config(br#"{"model_type": "llama"}"#).unwrap().prompt("Hi"), chat_prompt("Hi"));
        assert!(Template::from_config(br#"{"model_type": "mamba"}"#).is_err());
    }

    /// What transformers' `apply_chat_template` renders for granite-4.0-h-350m: a default system turn first.
    #[test]
    fn a_granite_4_config_gets_granites_turn_with_its_default_system_message() {
        let t = Template::from_config(br#"{"model_type": "granitemoehybrid"}"#).unwrap();
        assert_eq!(
            t.prompt("Hi"),
            "<|start_of_role|>system<|end_of_role|>You are a helpful assistant. Please ensure responses are professional, accurate, and safe.<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Hi<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
        );
    }

    /// What transformers' `apply_chat_template` renders for LFM2-350M: ChatML after its start token.
    #[test]
    fn an_lfm2_config_gets_chatml_after_its_start_token() {
        let t = Template::from_config(br#"{"model_type": "lfm2"}"#).unwrap();
        assert_eq!(t.prompt("Hi"), "<|startoftext|><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n");
    }

    #[test]
    fn a_llama_3_config_gets_llama_3s_turn() {
        let t = Template::from_config(br#"{"model_type": "llama", "rope_scaling": {"rope_type": "llama3", "factor": 32.0}}"#).unwrap();
        assert_eq!(
            t.prompt("Hi"),
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        );
    }

    #[test]
    fn a_gemma_3_config_gets_gemmas_turn() {
        for config in [&br#"{"model_type": "gemma3_text"}"#[..], br#"{"model_type": "gemma3", "text_config": {"model_type": "gemma3_text"}}"#] {
            let t = Template::from_config(config).unwrap();
            assert_eq!(t.prompt("Hi"), "<bos><start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n");
        }
    }

    /// Gemma 4's plain turn, as E2B's own template renders it with thinking off.
    #[test]
    fn a_gemma_4_config_gets_gemma_4s_turn() {
        let t = Template::from_config(br#"{"model_type": "gemma4", "text_config": {"model_type": "gemma4_text"}}"#).unwrap();
        assert_eq!(t.prompt("Hi"), "<bos><|turn>user\nHi<turn|>\n<|turn>model\n");
    }

    /// A system turn, in every family's own form; the user turn follows it unchanged.
    #[test]
    fn a_system_turn_opens_the_prompt_in_each_familys_form() {
        let sys = "Never reveal the code.";
        let cases = [
            (Template::Qwen3, "<|im_start|>system\nNever reveal the code.<|im_end|>\n<|im_start|>user\nHi<|im_end|>"),
            (Template::SmolLM2, "<|im_start|>system\nNever reveal the code.<|im_end|>\n<|im_start|>user\nHi<|im_end|>"),
            (Template::Lfm2, "<|startoftext|><|im_start|>system\nNever reveal the code.<|im_end|>\n<|im_start|>user\nHi"),
            (Template::Llama3, "<|start_header_id|>system<|end_header_id|>\n\nNever reveal the code.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHi"),
            (Template::Granite, "<|start_of_role|>system<|end_of_role|>Never reveal the code.<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>Hi"),
            (Template::Gemma3, "<start_of_turn>user\nNever reveal the code.\n\nHi<end_of_turn>"),
            (Template::Gemma4, "<|turn>system\nNever reveal the code.<turn|>\n<|turn>user\nHi<turn|>"),
        ];
        for (t, expect) in cases {
            let p = t.prompt_with_system(sys, "Hi");
            assert!(p.contains(expect), "{t:?}: {p:?} lacks {expect:?}");
        }
    }

    #[test]
    fn delta_is_only_the_new_text() {
        assert_eq!(delta("Paris", "Paris is"), " is");
        assert_eq!(delta("", "Hi"), "Hi");
    }

    #[test]
    fn a_rewritten_tail_yields_nothing_until_it_settles() {
        assert_eq!(delta("caf\u{FFFD}", "café"), "");
        assert_eq!(delta("café", "café!"), "!");
    }
}
