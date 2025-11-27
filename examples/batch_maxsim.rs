//! High-throughput batch MaxSim scoring.
//!
//! Score a query against many documents efficiently.
//! Useful for reranking large candidate sets.
//!
//! Run: `cargo run --example batch_maxsim`

use rank_refine::simd::{maxsim_batch, normalize_maxsim, softmax_scores, top_k_indices};

fn main() {
    // Query: 32 tokens × 128 dims (typical ColBERT)
    let query_tokens: Vec<Vec<f32>> = (0..32)
        .map(|i| {
            let mut v = vec![0.0f32; 128];
            v[i % 128] = 1.0; // One-hot for demo
            v
        })
        .collect();

    // Documents: 100 docs, each with 64-256 tokens
    let documents: Vec<Vec<Vec<f32>>> = (0..100)
        .map(|doc_idx| {
            let n_tokens = 64 + (doc_idx % 193); // Variable length
            (0..n_tokens)
                .map(|tok_idx| {
                    let mut v = vec![0.0f32; 128];
                    // Some structure: docs with similar idx share tokens
                    v[(doc_idx + tok_idx) % 128] = 0.8;
                    v[(doc_idx * 2 + tok_idx) % 128] = 0.6;
                    v
                })
                .collect()
        })
        .collect();

    println!("=== Batch MaxSim Scoring ===\n");
    println!("Query: {} tokens × {} dims", query_tokens.len(), 128);
    println!("Documents: {} docs", documents.len());

    // Score all documents
    let start = std::time::Instant::now();
    let scores = maxsim_batch(&query_tokens, &documents);
    let elapsed = start.elapsed();

    println!("\nScored {} documents in {:?}", documents.len(), elapsed);
    println!(
        "Throughput: {:.0} docs/sec",
        documents.len() as f64 / elapsed.as_secs_f64()
    );

    // Raw scores (sum of max similarities)
    println!("\nTop 5 raw scores:");
    let mut indexed: Vec<_> = scores.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (idx, score) in indexed.iter().take(5) {
        println!("  doc_{idx}: {score:.2}");
    }

    // Normalized to [0, 1] (divide by query length)
    let query_len = query_tokens.len() as u32;
    let normalized: Vec<f32> = scores
        .iter()
        .map(|&s| normalize_maxsim(s, query_len))
        .collect();

    println!("\nTop 5 normalized [0,1]:");
    let mut indexed: Vec<_> = normalized.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (idx, score) in indexed.iter().take(5) {
        println!("  doc_{idx}: {score:.4}");
    }

    // Softmax for probability distribution
    let probs = softmax_scores(&scores);

    println!("\nTop 5 softmax probabilities:");
    let mut indexed: Vec<_> = probs.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (idx, prob) in indexed.iter().take(5) {
        println!("  doc_{idx}: {:.2}%", *prob * 100.0);
    }

    // Efficient top-k selection
    let top_10 = top_k_indices(&scores, 10);

    println!("\nTop 10 indices (efficient selection):");
    println!("  {:?}", top_10);
}
