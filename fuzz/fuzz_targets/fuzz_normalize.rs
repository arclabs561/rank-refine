#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use rank_refine::simd;

#[derive(Arbitrary, Debug)]
struct NormalizeInput {
    scores: Vec<f32>,
    query_maxlen: u32,
    k: usize,
}

fuzz_target!(|input: NormalizeInput| {
    // Should not panic on any input

    // Normalize single score
    if input.query_maxlen > 0 {
        let _ = simd::normalize_maxsim(
            input.scores.first().copied().unwrap_or(0.0),
            input.query_maxlen,
        );
    }

    // Batch normalize
    if input.query_maxlen > 0 {
        let _ = simd::normalize_maxsim_batch(&input.scores, input.query_maxlen);
    }

    // Softmax
    let _ = simd::softmax_scores(&input.scores);

    // Top-k
    let _ = simd::top_k_indices(&input.scores, input.k);
});
