//! Upstream's TTFT demo prompt — `examples/ttft_demo.py::make_prompt` at `6be1956f`, ported so the
//! models read the same bytes upstream's harness gives them.

pub const DEFAULT_QUESTION: &str =
    "Identify the three cache failures in the log. Give one concrete fix for each. Be concise.";

/// The synthetic cache-incident log truncated to `count` source-tokenizer tokens, then the task.
/// Returns the user message and the token count of the log part.
pub fn incident_log(tokenizer_json: &str, count: usize, question: &str) -> Result<(String, usize), String> {
    let tok = tokenizers::Tokenizer::from_file(tokenizer_json).map_err(|e| format!("{tokenizer_json}: {e}"))?;
    let entries: Vec<String> = (0..count / 12 + 1)
        .map(|i| {
            let body = match i % 3 {
                0 => "catalog: cached price version 41 served after database advanced to version 42; invalidation event missing. ",
                1 => "checkout: hot key expired; 500 requests simultaneously queried the database; no request coalescing. ",
                _ => "session: cache memory at capacity; keys evicted repeatedly; cache misses and database latency rising. ",
            };
            format!("Incident sample {}: {body}", i + 1)
        })
        .collect();
    let text = format!("CACHE INCIDENT LOG (synthetic demonstration data)\n{}", entries.join("\n"));
    // codec.encode = tokenizer(text, add_special_tokens=False); codec.decode(ids, special=True).
    let ids = tok.encode(text, false).map_err(|e| e.to_string())?.get_ids().to_vec();
    let ids = &ids[..count.min(ids.len())];
    if ids.is_empty() {
        return Err("the context is empty".into());
    }
    let text = tok.decode(ids, false).map_err(|e| e.to_string())?;
    Ok((format!("{text}\n\nTASK: {question}"), ids.len()))
}
