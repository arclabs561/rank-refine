//! Example: Reranking with cosine similarity and Matryoshka refinement.
//!
//! Run: `cargo run --example rerank`

use rank_refine::simd::cosine;

fn main() {
    // Simulated embeddings (in real usage, get these from fastembed/candle/etc.)
    let query = vec![0.8, 0.2, 0.1, 0.05];
    let documents = vec![
        ("doc_relevant", vec![0.75, 0.25, 0.1, 0.1]),
        ("doc_partial", vec![0.4, 0.6, 0.2, 0.1]),
        ("doc_unrelated", vec![0.1, 0.1, 0.9, 0.8]),
    ];

    println!("Query embedding: {:?}", query);
    println!("Documents:");
    for (id, emb) in &documents {
        println!("  {id}: {:?}", emb);
    }

    // Score with cosine similarity
    println!("\n=== Cosine Similarity ===");
    let mut scored: Vec<_> = documents
        .iter()
        .map(|(id, emb)| (*id, cosine(&query, emb)))
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (id, score) in &scored {
        println!("  {id}: {score:.4}");
    }

    // Matryoshka refinement (use head dims first, then refine with tail)
    println!("\n=== Matryoshka Refinement ===");
    // 8-dim embeddings: head (first 4) + tail (last 4)
    let query_8d = vec![0.8, 0.2, 0.1, 0.05, 0.3, 0.1, 0.2, 0.1];
    let docs_8d: Vec<(&str, Vec<f32>)> = vec![
        ("doc_a", vec![0.75, 0.25, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1]),
        ("doc_b", vec![0.4, 0.6, 0.2, 0.1, 0.1, 0.2, 0.3, 0.2]),
        ("doc_c", vec![0.1, 0.1, 0.9, 0.8, 0.1, 0.1, 0.1, 0.1]),
    ];

    // Initial scores using first 4 dims (head)
    let initial: Vec<(&str, f32)> = docs_8d
        .iter()
        .map(|(id, emb)| (*id, cosine(&query_8d[..4], &emb[..4])))
        .collect();
    println!("Initial (4-dim head): {:?}", initial);

    // Refine using all 8 dims
    let refined = rank_refine::matryoshka::refine(&initial, &query_8d, &docs_8d, 4);
    println!("Refined (8-dim full): {:?}", refined);
}
