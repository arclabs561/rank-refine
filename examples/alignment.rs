//! Example: Token-level alignment and highlighting with ColBERT
//!
//! This demonstrates the interpretability features that distinguish ColBERT
//! from single-vector embeddings: showing which document tokens match each query token.

use rank_refine::prelude::*;
use rank_refine::simd::maxsim_vecs;

fn main() {
    // Query: "capital of France" (simplified to 2 tokens for demo)
    let query = vec![
        vec![1.0, 0.0, 0.0],  // "capital" token
        vec![0.0, 1.0, 0.0],  // "France" token
    ];

    // Document: "Paris is the capital of France" (simplified to 4 tokens)
    let doc = vec![
        vec![0.1, 0.1, 0.8],  // "Paris" token
        vec![0.2, 0.2, 0.6],  // "is" token
        vec![0.9, 0.1, 0.0],  // "capital" token (matches query[0])
        vec![0.1, 0.9, 0.0],  // "France" token (matches query[1])
    ];

    // Get token-level alignments
    let alignments = maxsim_alignments_vecs(&query, &doc);
    println!("Token alignments:");
    for (q_idx, d_idx, score) in &alignments {
        println!("  Query token {} → Doc token {} (similarity: {:.3})", q_idx, d_idx, score);
    }

    // Extract highlighted tokens (above threshold)
    let highlighted = highlight_matches_vecs(&query, &doc, 0.7);
    println!("\nHighlighted document token indices: {:?}", highlighted);
    println!("These tokens can be used for snippet extraction or highlighting in search results.");

    // Verify alignment sum equals MaxSim score
    let maxsim_score = maxsim_vecs(&query, &doc);
    let alignment_sum: f32 = alignments.iter().map(|(_, _, s)| s).sum();
    println!("\nMaxSim score: {:.3}", maxsim_score);
    println!("Sum of alignment scores: {:.3}", alignment_sum);
    assert!((maxsim_score - alignment_sum).abs() < 1e-5, "Should match!");
}

