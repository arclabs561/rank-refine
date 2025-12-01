//! Example: Batch alignment and highlighting for reranking pipelines
//!
//! Demonstrates how to efficiently get alignments and highlights for multiple documents
//! when reranking candidate sets.

use rank_refine::prelude::*;

fn main() {
    println!("Batch Alignment Example\n");

    // Query: "rust memory safety" (3 tokens, 128-dim each)
    let query = vec![
        vec![1.0, 0.0, 0.0, 0.0],  // "rust" token
        vec![0.0, 1.0, 0.0, 0.0],  // "memory" token
        vec![0.0, 0.0, 1.0, 0.0],  // "safety" token
    ];

    // Documents to rerank (simplified for demo)
    let documents = vec![
        vec![
            vec![0.9, 0.1, 0.0, 0.0],  // "rust" token
            vec![0.1, 0.9, 0.0, 0.0],  // "memory" token
            vec![0.0, 0.0, 0.9, 0.1],  // "safety" token
        ],
        vec![
            vec![0.5, 0.5, 0.0, 0.0],  // "programming" token
            vec![0.0, 0.5, 0.5, 0.0],  // "language" token
        ],
        vec![
            vec![0.9, 0.1, 0.0, 0.0],  // "rust" token
            vec![0.0, 0.0, 0.0, 0.0],  // filler
        ],
    ];

    println!("Query: {} tokens", query.len());
    println!("Documents: {} docs\n", documents.len());

    // 1. Batch scoring
    let scores = maxsim_batch(&query, &documents);
    println!("MaxSim scores:");
    for (i, &score) in scores.iter().enumerate() {
        println!("  Doc {}: {:.3}", i, score);
    }

    // 2. Batch alignments
    let all_alignments = maxsim_alignments_batch(&query, &documents);
    println!("\nAlignments per document:");
    for (doc_idx, alignments) in all_alignments.iter().enumerate() {
        println!("  Doc {}: {} alignments", doc_idx, alignments.len());
        for (q_idx, d_idx, score) in alignments.iter().take(3) {
            println!("    Query[{}] → Doc[{}]: {:.3}", q_idx, d_idx, score);
        }
    }

    // 3. Batch highlights
    let threshold = 0.7;
    let all_highlights = highlight_matches_batch(&query, &documents, threshold);
    println!("\nHighlighted tokens (threshold={}):", threshold);
    for (doc_idx, highlights) in all_highlights.iter().enumerate() {
        println!("  Doc {}: {:?}", doc_idx, highlights);
    }

    // 4. Use alignment utilities
    println!("\nAlignment utilities:");
    for (doc_idx, alignments) in all_alignments.iter().enumerate() {
        // Top-k alignments
        let top2 = top_k_alignments(alignments, 2);
        println!("  Doc {} top-2 alignments:", doc_idx);
        for (q_idx, d_idx, score) in &top2 {
            println!("    Query[{}] → Doc[{}]: {:.3}", q_idx, d_idx, score);
        }

        // Filter by threshold
        let filtered = filter_alignments(alignments, 0.8);
        println!("  Doc {} high-quality (>=0.8): {} alignments", doc_idx, filtered.len());

        // Statistics
        let (min, max, mean, sum, count) = alignment_stats(alignments);
        println!("  Doc {} stats: min={:.3}, max={:.3}, mean={:.3}, sum={:.3}, count={}",
                 doc_idx, min, max, mean, sum, count);
    }

    // 5. Verify consistency: alignment sum equals MaxSim
    println!("\nConsistency check:");
    for (doc_idx, alignments) in all_alignments.iter().enumerate() {
        let alignment_sum: f32 = alignments.iter().map(|(_, _, s)| s).sum();
        let maxsim_score = scores[doc_idx];
        assert!(
            (alignment_sum - maxsim_score).abs() < 1e-4,
            "Doc {}: Alignment sum should equal MaxSim",
            doc_idx
        );
        println!("  Doc {}: ✓ Alignment sum ({:.3}) = MaxSim ({:.3})",
                 doc_idx, alignment_sum, maxsim_score);
    }

    println!("\n✓ Batch alignment functions work correctly!");
}

