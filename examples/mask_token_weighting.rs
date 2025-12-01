//! Example: Using [MASK] tokens with weighted MaxSim
//!
//! ColBERT uses [MASK] tokens for query augmentation. These tokens are added during
//! encoding and act as soft query expansion, allowing the model to learn which
//! query terms should be emphasized.
//!
//! This example shows how to use `maxsim_weighted()` with [MASK] token embeddings
//! and their learned importance weights.

use rank_refine::prelude::*;
use rank_refine::simd::{maxsim_vecs, maxsim_weighted_vecs};

fn main() {
    println!("[MASK] Token Weighting Example\n");

    // Original query: "rust memory" (2 tokens)
    let original_query = vec![
        vec![1.0, 0.0, 0.0],  // "rust" token
        vec![0.0, 1.0, 0.0],  // "memory" token
    ];

    // Query with [MASK] tokens appended (ColBERT typically adds 2-4 [MASK] tokens)
    // In practice, these would come from the ColBERT encoder model
    let query_with_masks = vec![
        vec![1.0, 0.0, 0.0],  // "rust" token (original)
        vec![0.0, 1.0, 0.0],  // "memory" token (original)
        vec![0.3, 0.3, 0.4],  // [MASK] token 1 (learned embedding)
        vec![0.2, 0.4, 0.4],  // [MASK] token 2 (learned embedding)
    ];

    // Document: "rust programming language memory safety"
    let doc = vec![
        vec![0.9, 0.1, 0.0],  // "rust" token
        vec![0.1, 0.2, 0.7],  // "programming" token
        vec![0.1, 0.1, 0.8],  // "language" token
        vec![0.0, 0.9, 0.1],  // "memory" token
        vec![0.0, 0.1, 0.9],  // "safety" token
    ];

    println!("Original query: {} tokens", original_query.len());
    println!("Query with [MASK]: {} tokens", query_with_masks.len());
    println!("Document: {} tokens\n", doc.len());

    // Method 1: Unweighted (treats all tokens equally, including [MASK])
    let unweighted_score = maxsim_vecs(&query_with_masks, &doc);
    println!("Unweighted MaxSim (all tokens equal): {:.3}", unweighted_score);

    // Method 2: Weighted (emphasize original query tokens, de-emphasize [MASK])
    // Research shows [MASK] tokens typically have lower learned importance
    // Weights are typically learned during training or inferred from attention
    let weights = vec![
        1.0,  // "rust" - high importance
        1.0,  // "memory" - high importance
        0.3,  // [MASK] 1 - lower importance (soft expansion)
        0.3,  // [MASK] 2 - lower importance (soft expansion)
    ];

    let weighted_score = maxsim_weighted_vecs(&query_with_masks, &doc, &weights);
    println!("Weighted MaxSim ([MASK] de-emphasized): {:.3}", weighted_score);

    // Method 3: Learned weights from attention mechanism
    // In practice, ColBERT models learn attention weights for [MASK] tokens
    // during training. These can be extracted from the model's attention layers.
    let learned_weights = vec![
        1.0,   // "rust" - original token, full weight
        1.0,   // "memory" - original token, full weight
        0.25,  // [MASK] 1 - learned importance (from attention)
        0.15,  // [MASK] 2 - learned importance (from attention)
    ];

    let learned_score = maxsim_weighted_vecs(&query_with_masks, &doc, &learned_weights);
    println!("Learned weights (from attention): {:.3}", learned_score);

    // Compare with original query (no [MASK] tokens)
    let original_score = maxsim_vecs(&original_query, &doc);
    println!("\nOriginal query (no [MASK]): {:.3}", original_score);
    println!("Query with [MASK] (unweighted): {:.3}", unweighted_score);
    println!("Query with [MASK] (weighted): {:.3}", weighted_score);
    println!("Query with [MASK] (learned): {:.3}", learned_score);

    // Analysis: [MASK] tokens can improve recall by expanding the query
    // but should be weighted lower to maintain precision
    println!("\n✓ [MASK] tokens provide soft query expansion");
    println!("  - Unweighted: All tokens contribute equally (may over-expand)");
    println!("  - Weighted: Original tokens emphasized, [MASK] de-emphasized");
    println!("  - Learned: Use attention weights from trained model (best)");

    // Get alignments to see which tokens match
    let alignments = maxsim_alignments_vecs(&query_with_masks, &doc);
    println!("\nToken alignments:");
    for (q_idx, d_idx, score) in &alignments {
        let token_type = if *q_idx < original_query.len() {
            "original"
        } else {
            "[MASK]"
        };
        println!(
            "  Query[{}] ({}) → Doc[{}]: {:.3}",
            q_idx, token_type, d_idx, score
        );
    }
}

