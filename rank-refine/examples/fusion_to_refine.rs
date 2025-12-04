//! Example: Integration between rank-fusion and rank-refine.
//!
//! Demonstrates the complete pipeline: Retrieve → Fuse → Refine → Top-K
//!
//! **Note**: This is a **simulated** integration example.
//! 
//! For a real integration, add `rank-fusion` as a dependency:
//! ```toml
//! [dependencies]
//! rank-fusion = "0.1"
//! ```
//! 
//! Then uncomment the imports and replace simulated fusion with real calls:
//! ```rust
//! use rank_fusion::explain::{rrf_explain, RetrieverId};
//! use rank_fusion::RrfConfig;
//! ```
//!
//! Run: `cargo run --example fusion_to_refine`
use rank_refine::explain::{rerank_batch, Candidate, RerankMethod, RerankerInput};

fn main() {
    println!("=== Fusion → Refine Pipeline ===\n");
    println!("Note: This example requires rank-fusion as a dependency.");
    println!("See the example source for integration details.\n");

    // Step 1: Simulate retrieval results from multiple retrievers
    let bm25_results = vec![
        ("doc_123", 87.5),
        ("doc_456", 82.3),
        ("doc_789", 78.1),
        ("doc_101", 75.0),
    ];

    let dense_results = vec![
        ("doc_456", 0.92),
        ("doc_123", 0.88),
        ("doc_999", 0.85),
        ("doc_101", 0.80),
    ];

    println!("1. Retrieval Results:");
    println!(
        "   BM25: {:?}",
        bm25_results.iter().map(|(id, _)| *id).collect::<Vec<_>>()
    );
    println!(
        "   Dense: {:?}",
        dense_results.iter().map(|(id, _)| *id).collect::<Vec<_>>()
    );

    // Step 2: Simulate fused results (in real usage, use rank-fusion)
    // Uncomment when rank-fusion is available:
    // let retrievers = vec![RetrieverId::new("bm25"), RetrieverId::new("dense")];
    // let fused = rrf_explain(
    //     &[&bm25_results[..], &dense_results[..]],
    //     &retrievers,
    //     RrfConfig::default(),
    // );

    // For demo, simulate fused results
    let fused_scores = vec![
        ("doc_456", 0.0331),
        ("doc_123", 0.0330),
        ("doc_101", 0.0164),
        ("doc_789", 0.0164),
        ("doc_999", 0.0164),
    ];

    println!("\n2. Fused Results (simulated):");
    for (i, (id, score)) in fused_scores.iter().take(5).enumerate() {
        println!("   {}. {} (score: {:.6})", i + 1, id, score);
    }

    // Step 3: Convert to refine input
    // In a real system, you'd load token embeddings from your embedding store
    // For demo, we'll create mock token embeddings
    let mock_token_embeddings: std::collections::HashMap<&str, Vec<Vec<f32>>> = [
        ("doc_123", vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
        ("doc_456", vec![vec![0.9, 0.1], vec![0.1, 0.9]]),
        ("doc_789", vec![vec![0.7, 0.3], vec![0.3, 0.7]]),
        ("doc_101", vec![vec![0.8, 0.2], vec![0.2, 0.8]]),
        ("doc_999", vec![vec![0.6, 0.4], vec![0.4, 0.6]]),
    ]
    .into_iter()
    .collect();

    let query_tokens = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

    let candidates: Vec<Candidate<&str>> = fused_scores
        .iter()
        .take(10) // Top 10 from fusion
        .map(|(id, score)| Candidate {
            id: *id,
            original_score: *score,
            dense_embedding: None,
            token_embeddings: mock_token_embeddings
                .get(*id)
                .map(|tokens| tokens.as_slice()),
            text: None,
        })
        .collect();

    let refine_input = RerankerInput {
        query_dense: None,
        query_tokens: Some(&query_tokens),
        candidates,
    };

    // Step 4: Refine with MaxSim
    let refined = rerank_batch(refine_input, RerankMethod::MaxSim, 5);

    println!("\n3. Refined Results (MaxSim reranking):");
    for (i, result) in refined.iter().enumerate() {
        println!(
            "   {}. {} (refined: {:.6}, original: {:.6}, rank change: {:+})",
            i + 1,
            result.id,
            result.score,
            result.original_score,
            result.rank as i32 - i as i32
        );
    }

    println!("\n=== Pipeline Complete ===");
    println!("\nKey insights:");
    println!("- Fusion finds consensus across retrievers");
    println!("- Refinement re-ranks using late interaction (MaxSim)");
    println!("- Final ranking may differ from fusion due to token-level matching");
}
