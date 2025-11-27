//! MMR Diversity
//!
//! Select diverse results from a similarity matrix.
//!
//! ```text
//! λ=1.0 → pure relevance
//! λ=0.5 → balanced (default)
//! λ=0.0 → pure diversity
//! ```
//!
//! Run: `cargo run --example search_diversity`

use rank_refine::diversity::{mmr, MmrConfig};
use rank_refine::simd::cosine;

fn main() {
    // Candidates: (id, relevance_score)
    let candidates = vec![
        ("doc_a", 0.95),
        ("doc_b", 0.90),  // similar to doc_a
        ("doc_c", 0.85),
        ("doc_d", 0.80),  // similar to doc_c
        ("doc_e", 0.75),  // unique
    ];

    // Embeddings for similarity computation
    let embeddings: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0],  // doc_a
        vec![0.9, 0.1, 0.0],  // doc_b: similar to a
        vec![0.0, 1.0, 0.0],  // doc_c
        vec![0.1, 0.9, 0.0],  // doc_d: similar to c
        vec![0.0, 0.0, 1.0],  // doc_e: unique direction
    ];

    // Build similarity matrix
    let n = embeddings.len();
    let mut similarity = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            similarity[i * n + j] = cosine(&embeddings[i], &embeddings[j]);
        }
    }

    // Compare lambda values
    for lambda in [1.0, 0.5, 0.0] {
        let config = MmrConfig::default().with_lambda(lambda).with_k(3);
        let selected = mmr(&candidates, &similarity, config);

        let ids: Vec<_> = selected.iter().map(|(id, _)| *id).collect();
        println!("λ={:.1}: {:?}", lambda, ids);
    }
}
