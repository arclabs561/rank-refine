//! Two-stage retrieval with Matryoshka embeddings.
//!
//! Stage 1: Fast ANN search using first 64 dims (coarse)
//! Stage 2: Refine using remaining dims (fine-grained)
//!
//! Run: `cargo run --example matryoshka_search`

use rank_refine::matryoshka;

fn main() {
    // Simulated: Your vector DB searched using head dims (0..64)
    // and returned these candidates with their coarse scores
    let candidates = vec![
        ("doc_rust_ownership", 0.92),
        ("doc_rust_borrowing", 0.91),
        ("doc_cpp_memory", 0.89),
        ("doc_rust_lifetimes", 0.88),
        ("doc_gc_languages", 0.85),
    ];

    // Full 128-dim embeddings (head=64, tail=64)
    // In practice: fetch these from your vector DB by ID
    let query = generate_embedding("How does Rust handle memory safety?");

    let docs = vec![
        (
            "doc_rust_ownership",
            generate_embedding("Rust ownership system"),
        ),
        (
            "doc_rust_borrowing",
            generate_embedding("Rust borrowing rules"),
        ),
        ("doc_cpp_memory", generate_embedding("C++ manual memory")),
        (
            "doc_rust_lifetimes",
            generate_embedding("Rust lifetime annotations"),
        ),
        (
            "doc_gc_languages",
            generate_embedding("Garbage collected languages"),
        ),
    ];

    println!("=== Two-Stage Matryoshka Search ===\n");
    println!("Stage 1 (coarse, head 64 dims):");
    for (id, score) in &candidates {
        println!("  {id}: {score:.2}");
    }

    // Stage 2: Refine using tail dimensions
    let head_dims = 64;

    // Default blend (50% coarse, 50% tail)
    let refined = matryoshka::refine(&candidates, &query, &docs, head_dims);

    println!("\nStage 2 (refined, using tail dims):");
    for (id, score) in &refined {
        println!("  {id}: {score:.4}");
    }

    // Tail-only (ignore coarse scores entirely)
    let tail_only = matryoshka::refine_tail_only(&candidates, &query, &docs, head_dims);

    println!("\nTail-only refinement:");
    for (id, score) in tail_only.iter().take(3) {
        println!("  {id}: {score:.4}");
    }

    // Custom alpha (80% coarse, 20% tail)
    let conservative = matryoshka::refine_with_alpha(&candidates, &query, &docs, head_dims, 0.8);

    println!("\nConservative (80% coarse, 20% tail):");
    for (id, score) in conservative.iter().take(3) {
        println!("  {id}: {score:.4}");
    }
}

/// Simulate embedding generation (deterministic for demo)
fn generate_embedding(text: &str) -> Vec<f32> {
    let mut emb = vec![0.0f32; 128];
    for (i, byte) in text.bytes().enumerate() {
        emb[i % 128] += (byte as f32) / 1000.0;
    }
    // Normalize
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut emb {
            *x /= norm;
        }
    }
    emb
}
