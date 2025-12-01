#![no_main]

use libfuzzer_sys::fuzz_target;
use rank_refine::colbert;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }

    let mut offset = 0;
    let n_tokens = (data[offset] as usize % 32) + 1; // 1-32 tokens
    offset += 1;
    let dim = (data[offset] as usize % 64) + 1; // 1-64 dimensions
    offset += 1;
    let pool_factor = (data[offset] as usize % 5) + 1; // 1-5 pool factor
    offset += 1;

    let required_bytes = n_tokens * dim * 4;
    if data.len() < offset + required_bytes {
        return;
    }

    // Build token vectors from fuzz input
    let mut tokens = Vec::with_capacity(n_tokens);
    for _ in 0..n_tokens {
        let mut token = Vec::with_capacity(dim);
        for _ in 0..dim {
            let val = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
            // Skip if NaN or infinite (these are handled but slow down fuzzing)
            if !val.is_finite() {
                return;
            }
            token.push(val);
            offset += 4;
        }
        tokens.push(token);
    }

    // Test all pooling methods - they should not panic
    let _ = colbert::pool_tokens(&tokens, pool_factor);
    let _ = colbert::pool_tokens_sequential(&tokens, pool_factor);
    let _ = colbert::pool_tokens_adaptive(&tokens, pool_factor);

    // Test with protected tokens (if we have enough)
    if n_tokens > 1 {
        let protected = (data.get(offset).copied().unwrap_or(0) as usize) % n_tokens;
        let _ = colbert::pool_tokens_with_protected(&tokens, pool_factor, protected);
    }
});
