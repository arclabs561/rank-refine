//! Cross-encoder reranking for final precision.
//!
//! Cross-encoders jointly encode query + document for maximum accuracy.
//! Use after dense/MaxSim to rerank top 10-50 candidates.
//!
//! Run: `cargo run --example cross_encoder_rerank`

use rank_refine::crossencoder::{refine, rerank, CrossEncoderModel};

/// Mock cross-encoder (in practice: use ort/candle with a real model)
struct MockCrossEncoder;

impl CrossEncoderModel for MockCrossEncoder {
    fn score_batch(&self, query: &str, documents: &[&str]) -> Vec<f32> {
        // Simulate: score by word overlap (real model would use transformer)
        let query_lower = query.to_lowercase();
        let query_words: std::collections::HashSet<&str> = query_lower.split_whitespace().collect();

        documents
            .iter()
            .map(|doc| {
                let doc_lower = doc.to_lowercase();
                let doc_words: std::collections::HashSet<&str> =
                    doc_lower.split_whitespace().collect();
                let overlap = query_words
                    .iter()
                    .filter(|w| doc_words.contains(*w))
                    .count();
                overlap as f32 / query_words.len().max(1) as f32
            })
            .collect()
    }
}

fn main() {
    let model = MockCrossEncoder;

    // Candidates from Stage 1 (dense/MaxSim retrieval)
    let candidates = vec![
        ("doc1", "Rust memory safety through ownership"),
        ("doc2", "Python garbage collection and memory"),
        ("doc3", "Memory management in systems programming"),
        ("doc4", "Safety features in modern languages"),
        ("doc5", "Rust borrow checker prevents data races"),
    ];

    let query = "rust memory safety";

    println!("=== Cross-Encoder Reranking ===\n");
    println!("Query: \"{query}\"\n");

    println!("Before (retrieval order):");
    for (id, text) in &candidates {
        println!("  {id}: \"{text}\"");
    }

    // Pure cross-encoder reranking
    let reranked = rerank(&model, query, &candidates);

    println!("\nAfter cross-encoder reranking:");
    for (id, score) in &reranked {
        println!("  {id}: {score:.3}");
    }

    // Blend with original scores (e.g., from dense retrieval)
    let candidates_with_scores: Vec<_> = vec![
        ("doc1", "Rust memory safety through ownership", 0.95),
        ("doc2", "Python garbage collection and memory", 0.90),
        ("doc3", "Memory management in systems programming", 0.88),
        ("doc4", "Safety features in modern languages", 0.85),
        ("doc5", "Rust borrow checker prevents data races", 0.82),
    ];

    // alpha=0.3 means 30% original score, 70% cross-encoder
    let blended = refine(&model, query, &candidates_with_scores, 0.3);

    println!("\nBlended (30% dense + 70% cross-encoder):");
    for (id, score) in &blended {
        println!("  {id}: {score:.3}");
    }
}
