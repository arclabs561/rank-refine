//! ColBERT Token Pooling
//!
//! Reduce storage for multi-vector embeddings by clustering tokens.
//!
//! ```text
//! 128 tokens × 128 dims = 64KB per doc
//! pool_factor=4 → 32 tokens = 16KB per doc (75% smaller)
//! ```
//!
//! Run: `cargo run --example colbert_pooling`

use rank_refine::colbert::{pool_tokens, pool_tokens_sequential};
use rank_refine::simd::maxsim_vecs;

fn main() {
    // Simulated ColBERT doc tokens (128 dims each)
    let doc_tokens: Vec<Vec<f32>> = (0..64)
        .map(|i| {
            let mut v = vec![0.0; 128];
            v[i % 128] = 1.0;
            v[(i + 1) % 128] = 0.5;
            v
        })
        .collect();

    let query_tokens: Vec<Vec<f32>> = (0..8)
        .map(|i| {
            let mut v = vec![0.0; 128];
            v[i * 8] = 1.0;
            v
        })
        .collect();

    println!("Original: {} tokens", doc_tokens.len());

    // Pool with different methods
    let pooled_cluster = pool_tokens(&doc_tokens, 4);
    let pooled_seq = pool_tokens_sequential(&doc_tokens, 4);

    println!("Clustered: {} tokens", pooled_cluster.len());
    println!("Sequential: {} tokens", pooled_seq.len());

    // Compare MaxSim scores
    let score_orig = maxsim_vecs(&query_tokens, &doc_tokens);
    let score_cluster = maxsim_vecs(&query_tokens, &pooled_cluster);
    let score_seq = maxsim_vecs(&query_tokens, &pooled_seq);

    println!("\nMaxSim scores:");
    println!("  Original:   {:.4}", score_orig);
    println!(
        "  Clustered:  {:.4} ({:+.1}%)",
        score_cluster,
        (score_cluster / score_orig - 1.0) * 100.0
    );
    println!(
        "  Sequential: {:.4} ({:+.1}%)",
        score_seq,
        (score_seq / score_orig - 1.0) * 100.0
    );

    // Storage savings
    let bytes_orig = doc_tokens.len() * 128 * 4;
    let bytes_pooled = pooled_cluster.len() * 128 * 4;
    println!(
        "\nStorage: {}KB → {}KB ({:.0}% reduction)",
        bytes_orig / 1024,
        bytes_pooled / 1024,
        (1.0 - bytes_pooled as f64 / bytes_orig as f64) * 100.0
    );
}
