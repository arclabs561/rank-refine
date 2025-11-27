//! Diversity-aware result selection.
//!
//! Avoid returning near-duplicate results by balancing
//! relevance with diversity using MMR or DPP.
//!
//! Run: `cargo run --example diversity_selection`

use rank_refine::diversity::{dpp, mmr_cosine, DppConfig, MmrConfig};
use rank_refine::simd::cosine;

fn main() {
    // Search results with embeddings (after dense retrieval)
    // Note: embeddings should be normalized for cosine similarity
    let results = vec![
        (
            "python_async_intro",
            embed("python async await tutorial introduction"),
        ),
        (
            "python_asyncio_guide",
            embed("python asyncio guide comprehensive"),
        ),
        (
            "python_async_patterns",
            embed("python async patterns best practices"),
        ),
        ("rust_async_intro", embed("rust async await futures tokio")),
        ("js_promises", embed("javascript promises async await")),
        ("go_goroutines", embed("go goroutines concurrency channels")),
        (
            "python_threading",
            embed("python threading multiprocessing"),
        ),
        (
            "java_concurrency",
            embed("java concurrency executors threads"),
        ),
    ];

    // Relevance scores from retrieval (higher = more relevant)
    let scores: Vec<f32> = vec![0.95, 0.93, 0.91, 0.85, 0.82, 0.78, 0.75, 0.70];

    // Combine into candidates
    let candidates: Vec<_> = results
        .iter()
        .zip(scores.iter())
        .map(|((id, _), &score)| (*id, score))
        .collect();

    // Extract embeddings for diversity calculation
    let embeddings: Vec<&[f32]> = results.iter().map(|(_, e)| e.as_slice()).collect();

    println!("=== Diversity Selection ===\n");

    // Without diversity (pure relevance)
    println!("Top 4 by relevance only:");
    for (id, score) in candidates.iter().take(4) {
        println!("  {id}: {score:.2}");
    }

    // Check similarity between top results
    println!("\nSimilarity matrix (top 4):");
    for i in 0..4 {
        for j in 0..4 {
            let sim = cosine(&embeddings[i], &embeddings[j]);
            print!("  {:.2}", sim);
        }
        println!("  <- {}", candidates[i].0);
    }
    println!("  ^ Note: top 3 are very similar (all Python async)");

    // MMR with lambda=0.5 (balanced)
    let mmr_config = MmrConfig::default().with_lambda(0.5).with_k(4);
    let mmr_results = mmr_cosine(&candidates, &embeddings, mmr_config);

    println!("\nMMR (lambda=0.5, balanced):");
    for (id, score) in &mmr_results {
        println!("  {id}: {score:.3}");
    }

    // MMR with lambda=0.3 (more diversity)
    let mmr_diverse = MmrConfig::default().with_lambda(0.3).with_k(4);
    let mmr_diverse_results = mmr_cosine(&candidates, &embeddings, mmr_diverse);

    println!("\nMMR (lambda=0.3, more diverse):");
    for (id, score) in &mmr_diverse_results {
        println!("  {id}: {score:.3}");
    }

    // DPP (determinantal point process) - build similarity matrix
    let n = embeddings.len();
    let mut sim_matrix = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in 0..n {
            sim_matrix[i][j] = cosine(&embeddings[i], &embeddings[j]);
        }
    }

    let dpp_config = DppConfig::default().with_alpha(0.5).with_k(4);
    let dpp_results = dpp(&candidates, &sim_matrix, dpp_config);

    println!("\nDPP (alpha=0.5):");
    for (id, score) in &dpp_results {
        println!("  {id}: {score:.3}");
    }

    println!("\n=== Key Insight ===");
    println!("Pure relevance: 3 Python async docs (redundant)");
    println!("With diversity: Mix of Python, Rust, JS, Go (useful variety)");
}

/// Simple embedding simulation (hash-based, deterministic)
fn embed(text: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; 64];
    for (i, word) in text.split_whitespace().enumerate() {
        for (j, ch) in word.chars().enumerate() {
            let idx = ((ch as usize) * 7 + i * 13 + j * 17) % 64;
            v[idx] += 0.3;
        }
    }
    // Normalize
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}
