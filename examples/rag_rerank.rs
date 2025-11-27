//! RAG Reranking
//!
//! Rerank retrieved chunks before sending to LLM context window.
//!
//! ```text
//! qdrant.search(20) → rerank → top 5 → LLM prompt
//! ```
//!
//! Run: `cargo run --example rag_rerank`

use rank_refine::scoring::DenseScorer;

fn main() {
    // Simulated qdrant results (id, embedding, initial_score)
    let retrieved: Vec<(&str, Vec<f32>, f32)> = vec![
        ("chunk_1", mock_embed("Rust ownership and borrowing rules"), 0.89),
        ("chunk_2", mock_embed("Memory safety without garbage collection"), 0.87),
        ("chunk_3", mock_embed("Python list comprehensions"), 0.85),
        ("chunk_4", mock_embed("Rust lifetimes ensure references are valid"), 0.84),
        ("chunk_5", mock_embed("JavaScript async await patterns"), 0.82),
    ];

    let query_emb = mock_embed("How does Rust handle memory safety?");

    // Rerank with cosine similarity
    let scorer = DenseScorer::Cosine;

    let mut reranked: Vec<_> = retrieved
        .iter()
        .map(|(id, emb, _)| (*id, scorer.score(&query_emb, emb)))
        .collect();

    reranked.sort_by(|a, b| b.1.total_cmp(&a.1));

    println!("Query: \"How does Rust handle memory safety?\"\n");
    println!("Before: {:?}", retrieved.iter().map(|(id, _, _)| *id).collect::<Vec<_>>());
    println!("After:  {:?}", reranked.iter().map(|(id, _)| *id).collect::<Vec<_>>());

    // Take top-k for LLM context
    let context_chunks: Vec<_> = reranked.iter().take(3).collect();
    println!("\nContext for LLM: {:?}", context_chunks);
}

fn mock_embed(text: &str) -> Vec<f32> {
    let mut emb = vec![0.0; 64];
    for (i, b) in text.bytes().enumerate() {
        emb[i % 64] += b as f32 / 1000.0;
    }
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    emb.iter_mut().for_each(|x| *x /= norm);
    emb
}
