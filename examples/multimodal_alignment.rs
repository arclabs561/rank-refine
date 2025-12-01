//! Example: Multimodal alignment with ColPali-style image patches
//!
//! Demonstrates how token-level alignment works for vision-language retrieval.
//! In ColPali, document images are split into patches (e.g., 32×32 = 1024 patches),
//! and query text tokens align with these image patch embeddings.

use rank_refine::prelude::*;
use rank_refine::simd::maxsim_vecs;

fn main() {
    println!("Multimodal Alignment Example (ColPali-style)\n");
    println!("In ColPali, document images are split into patches (e.g., 32×32 grid = 1024 patches).");
    println!("Each patch becomes a 'token' embedding, and query text tokens align with patches.\n");

    // Query: "revenue Q3 chart" (3 text tokens)
    let query_tokens = vec![
        vec![1.0, 0.0, 0.0],  // "revenue" token
        vec![0.0, 1.0, 0.0],  // "Q3" token
        vec![0.0, 0.0, 1.0],  // "chart" token
    ];

    // Document: Image split into 4 patches (simplified for demo)
    // In real ColPali, this would be 1024 patches from a 32×32 grid
    let image_patches = vec![
        vec![0.1, 0.1, 0.8],  // Patch 0: mostly background
        vec![0.9, 0.1, 0.0],  // Patch 1: contains "revenue" text/visual
        vec![0.1, 0.9, 0.0],  // Patch 2: contains "Q3" label
        vec![0.0, 0.0, 0.9],  // Patch 3: contains chart visualization
    ];

    println!("Query: 'revenue Q3 chart' (3 text tokens)");
    println!("Document: Image with 4 patches\n");

    // Get alignments: which image patches match which query tokens?
    let alignments = maxsim_alignments_vecs(&query_tokens, &image_patches);
    println!("Token-to-patch alignments:");
    for (q_idx, patch_idx, score) in &alignments {
        let token = match q_idx {
            0 => "revenue",
            1 => "Q3",
            2 => "chart",
            _ => "unknown",
        };
        println!(
            "  Query token '{}' (idx {}) → Image patch {} (similarity: {:.3})",
            token, q_idx, patch_idx, score
        );
    }

    // Extract highlighted patches for visual snippet extraction
    let highlighted_patches = highlight_matches_vecs(&query_tokens, &image_patches, 0.7);
    println!("\nHighlighted image patch indices: {:?}", highlighted_patches);
    println!("These patches can be extracted as visual snippets to show users which");
    println!("regions of the document image are relevant to their query.");

    // Verify MaxSim score equals sum of alignment scores
    let maxsim_score = maxsim_vecs(&query_tokens, &image_patches);
    let alignment_sum: f32 = alignments.iter().map(|(_, _, s)| s).sum();
    println!("\nMaxSim score: {:.3}", maxsim_score);
    println!("Sum of alignment scores: {:.3}", alignment_sum);
    assert!(
        (maxsim_score - alignment_sum).abs() < 1e-5,
        "Should match!"
    );

    println!("\n✓ Multimodal alignment works identically to text-only alignment!");
    println!("  The same MaxSim mechanism enables both text-text and text-image retrieval.");
}

