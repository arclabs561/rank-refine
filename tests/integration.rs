//! Integration tests simulating realistic e2e workflows.
//!
//! These tests use synthetic embeddings that mimic real model outputs.
//! They verify the full pipeline works correctly without requiring actual models.

use rank_refine::{
    colbert, matryoshka, simd,
    scoring::{
        AdaptivePooler, ClusteringPooler, DenseScorer, LateInteractionScorer, Pooler, Scorer,
        SequentialPooler, TokenScorer, blend, normalize_scores,
    },
    crossencoder::CrossEncoderModel,
    RefineConfig,
};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Realistic Embedding Generators (simulate model outputs)
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a normalized embedding from text (deterministic hash-based).
fn mock_dense_embed(text: &str, dim: usize) -> Vec<f32> {
    let mut embedding = vec![0.0; dim];
    for (i, c) in text.chars().enumerate() {
        embedding[i % dim] += (c as u32 as f32) / 1000.0;
    }
    // L2 normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        embedding.iter_mut().for_each(|x| *x /= norm);
    }
    embedding
}

/// Generate token embeddings (one per word, simplified).
fn mock_token_embed(text: &str, dim: usize) -> Vec<Vec<f32>> {
    text.split_whitespace()
        .map(|word| mock_dense_embed(word, dim))
        .collect()
}

/// Mock cross-encoder that scores by word overlap.
struct OverlapEncoder;

impl CrossEncoderModel for OverlapEncoder {
    fn score_batch(&self, query: &str, documents: &[&str]) -> Vec<f32> {
        let query_lower = query.to_lowercase();
        let query_words: std::collections::HashSet<&str> = 
            query_lower.split_whitespace().collect();
        
        documents.iter().map(|doc| {
            let doc_lower = doc.to_lowercase();
            let doc_words: std::collections::HashSet<&str> = 
                doc_lower.split_whitespace().collect();
            let overlap = doc_words.iter().filter(|w| query_words.contains(*w)).count();
            overlap as f32 / query_words.len().max(1) as f32
        }).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: Two-Stage Dense → ColBERT Retrieval
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_two_stage_dense_then_colbert() {
    const DIM: usize = 128;
    
    // Corpus
    let documents = vec![
        ("d1", "Rust is a systems programming language focused on safety"),
        ("d2", "Python is great for data science and machine learning"),
        ("d3", "Rust guarantees memory safety without garbage collection"),
        ("d4", "JavaScript runs in web browsers and Node.js"),
    ];
    
    let query = "memory safe systems programming";
    
    // Stage 1: Dense retrieval
    let query_dense = mock_dense_embed(query, DIM);
    let docs_dense: Vec<_> = documents.iter()
        .map(|(id, text)| (*id, mock_dense_embed(text, DIM)))
        .collect();
    
    let scorer = DenseScorer::Cosine;
    let doc_refs: Vec<_> = docs_dense.iter()
        .map(|(id, emb)| (*id, emb.as_slice()))
        .collect();
    
    let first_stage = scorer.rank(&query_dense, &doc_refs);
    
    // Verify we got all documents
    assert_eq!(first_stage.len(), 4);
    
    // Stage 2: ColBERT refinement on top-2
    let top_2: Vec<_> = first_stage.iter().take(2).cloned().collect();
    
    let query_tokens = mock_token_embed(query, DIM);
    let doc_tokens: HashMap<_, _> = documents.iter()
        .map(|(id, text)| (*id, mock_token_embed(text, DIM)))
        .collect();
    
    let doc_token_vec: Vec<_> = top_2.iter()
        .filter_map(|(id, _)| doc_tokens.get(id).map(|t| (*id, t.clone())))
        .collect();
    
    let refined = colbert::refine(&top_2, &query_tokens, &doc_token_vec, 0.5);
    
    // Verify refinement works
    assert!(!refined.is_empty());
    assert!(refined.len() <= 2);
    
    // Scores should be sorted descending
    for window in refined.windows(2) {
        assert!(window[0].1 >= window[1].1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: Matryoshka Refinement
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_matryoshka_refinement() {
    const FULL_DIM: usize = 768;
    const HEAD_DIM: usize = 256;
    
    let documents = vec![
        ("d1", "Rust programming language"),
        ("d2", "Python scripting"),
        ("d3", "Rust memory safety"),
    ];
    
    let query = "Rust systems programming";
    
    // Generate full-dimension embeddings
    let query_emb = mock_dense_embed(query, FULL_DIM);
    let docs: Vec<_> = documents.iter()
        .map(|(id, text)| (*id, mock_dense_embed(text, FULL_DIM)))
        .collect();
    
    // Initial ranking with head dimensions only
    let head_scorer = DenseScorer::Cosine;
    let head_docs: Vec<_> = docs.iter()
        .map(|(id, emb)| (*id, &emb[..HEAD_DIM]))
        .collect();
    
    let initial_ranking = head_scorer.rank(&query_emb[..HEAD_DIM], &head_docs);
    
    // Refine using tail dimensions
    let candidates: Vec<_> = initial_ranking.iter()
        .map(|(id, score)| (*id, *score))
        .collect();
    
    let refined = matryoshka::refine(&candidates, &query_emb, &docs, HEAD_DIM);
    
    // Verify
    assert_eq!(refined.len(), documents.len());
    for window in refined.windows(2) {
        assert!(window[0].1 >= window[1].1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: Token Pooling for Storage Efficiency
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_token_pooling_storage_workflow() {
    const DIM: usize = 128;
    
    // Simulate a long document with many tokens
    let long_doc = "Rust is a multi-paradigm systems programming language \
                    focused on safety especially safe concurrency Rust is \
                    syntactically similar to C++ but provides memory safety \
                    without using garbage collection";
    
    let original_tokens = mock_token_embed(long_doc, DIM);
    let original_count = original_tokens.len();
    
    // Test different pooling strategies
    let poolers: Vec<(&str, Box<dyn Pooler>)> = vec![
        ("sequential", Box::new(SequentialPooler)),
        ("clustering", Box::new(ClusteringPooler)),
        ("adaptive", Box::new(AdaptivePooler)),
    ];
    
    for (name, pooler) in poolers {
        // Pool with factor 2 (50% reduction)
        let pooled = pooler.pool_by_factor(&original_tokens, 2);
        
        // Verify reduction
        assert!(pooled.len() <= original_count, "{} should reduce count", name);
        assert!(pooled.len() >= original_count / 3, "{} shouldn't over-reduce", name);
        
        // Verify dimension preserved
        for token in &pooled {
            assert_eq!(token.len(), DIM, "{} should preserve dimension", name);
        }
        
        // Verify pooled tokens can still be scored
        let query_tokens = mock_token_embed("Rust memory safety", DIM);
        let score_original = simd::maxsim_vecs(&query_tokens, &original_tokens);
        let score_pooled = simd::maxsim_vecs(&query_tokens, &pooled);
        
        // Pooled score should be in reasonable range of original
        assert!(score_pooled > 0.0, "{} should produce positive score", name);
        // Allow up to 50% degradation (pooling is lossy)
        assert!(
            score_pooled >= score_original * 0.5,
            "{}: pooled score {} too far from original {}",
            name, score_pooled, score_original
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: Cross-Encoder Reranking
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_cross_encoder_rerank() {
    let query = "rust memory safety";
    let documents = vec![
        "Rust guarantees memory safety without garbage collection",
        "Python is dynamically typed",
        "Memory management in Rust is automatic and safe",
    ];
    
    let model = OverlapEncoder;
    
    // Initial scores (simulating first-stage retrieval)
    let _initial: Vec<_> = documents.iter()
        .enumerate()
        .map(|(i, _)| (i, 0.5 + i as f32 * 0.1)) // arbitrary initial scores
        .collect();
    
    // Rerank with cross-encoder
    use rank_refine::crossencoder::rerank;
    let doc_refs: Vec<_> = documents.iter()
        .enumerate()
        .map(|(i, s)| (i, *s))
        .collect();
    let reranked = rerank(&model, query, &doc_refs);
    
    // Documents with "memory" and "rust" should rank higher
    assert_eq!(reranked.len(), 3);
    // Doc 0 and 2 mention "memory" and related terms
    let top_id = reranked[0].0;
    assert!(top_id == 0 || top_id == 2, "Expected doc with overlap to rank first");
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: Hybrid Scoring with Score Normalization
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_hybrid_scoring_with_normalization() {
    const DIM: usize = 64;
    
    let query = "fast systems programming";
    let documents = vec![
        ("d1", "Rust is blazingly fast"),
        ("d2", "Python is slow but easy"),
        ("d3", "C++ is fast but unsafe"),
    ];
    
    // Dense scores
    let query_emb = mock_dense_embed(query, DIM);
    let dense_scores: Vec<f32> = documents.iter()
        .map(|(_, text)| {
            let doc_emb = mock_dense_embed(text, DIM);
            simd::cosine(&query_emb, &doc_emb)
        })
        .collect();
    
    // BM25-like scores (simulated)
    let bm25_scores: Vec<f32> = vec![12.5, 3.2, 8.7];
    
    // Normalize both
    let dense_norm = normalize_scores(&dense_scores);
    let bm25_norm = normalize_scores(&bm25_scores);
    
    // Hybrid: 70% dense, 30% BM25
    let hybrid: Vec<f32> = dense_norm.iter()
        .zip(&bm25_norm)
        .map(|(d, b)| blend(*d, *b, 0.7))
        .collect();
    
    // Verify normalization worked
    for &score in &dense_norm {
        assert!(score >= 0.0 && score <= 1.0, "Normalized score out of bounds");
    }
    for &score in &bm25_norm {
        assert!(score >= 0.0 && score <= 1.0, "Normalized score out of bounds");
    }
    
    // Verify hybrid scores are reasonable
    assert_eq!(hybrid.len(), 3);
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: Full Pipeline with Config
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_full_pipeline_with_config() {
    const DIM: usize = 128;
    
    let query = "machine learning with rust";
    let documents: Vec<(&str, &str)> = vec![
        ("d1", "Rust for machine learning is growing"),
        ("d2", "Python dominates ML ecosystem"),
        ("d3", "Rust ML frameworks like candle"),
        ("d4", "JavaScript for web development"),
        ("d5", "Rust systems programming"),
    ];
    
    // Generate embeddings
    let query_tokens = mock_token_embed(query, DIM);
    let doc_data: Vec<_> = documents.iter()
        .map(|(id, text)| (*id, mock_token_embed(text, DIM)))
        .collect();
    
    // Initial ranking
    let initial: Vec<_> = colbert::rank(&query_tokens, &doc_data);
    
    // Refine with config
    let config = RefineConfig::default()
        .with_alpha(0.6)
        .with_top_k(3);
    
    let refined = colbert::refine_with_config(
        &initial,
        &query_tokens,
        &doc_data,
        config,
    );
    
    // Verify config respected
    assert!(refined.len() <= 3, "top_k should limit results");
    
    // Verify sorted
    for window in refined.windows(2) {
        assert!(window[0].1 >= window[1].1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: Trait-Based Polymorphism
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_trait_based_scoring() {
    const DIM: usize = 64;
    
    // Generic function that works with any Scorer
    fn score_docs<S: Scorer>(scorer: &S, query: &[f32], docs: &[&[f32]]) -> Vec<f32> {
        docs.iter().map(|d| scorer.score(query, d)).collect()
    }
    
    // Generic function that works with any TokenScorer
    fn score_token_docs<S: TokenScorer>(
        scorer: &S, 
        query: &[Vec<f32>], 
        docs: &[Vec<Vec<f32>>]
    ) -> Vec<f32> {
        docs.iter().map(|d| scorer.score_vecs(query, d)).collect()
    }
    
    let query = mock_dense_embed("test query", DIM);
    let docs: Vec<Vec<f32>> = vec![
        mock_dense_embed("doc one", DIM),
        mock_dense_embed("doc two", DIM),
    ];
    let doc_refs: Vec<&[f32]> = docs.iter().map(|d| d.as_slice()).collect();
    
    // Test with different scorers
    let dot_scores = score_docs(&DenseScorer::Dot, &query, &doc_refs);
    let cosine_scores = score_docs(&DenseScorer::Cosine, &query, &doc_refs);
    
    assert_eq!(dot_scores.len(), 2);
    assert_eq!(cosine_scores.len(), 2);
    
    // Test with token scorers
    let query_tokens = mock_token_embed("test query", DIM);
    let doc_tokens: Vec<Vec<Vec<f32>>> = vec![
        mock_token_embed("doc one text", DIM),
        mock_token_embed("doc two text", DIM),
    ];
    
    let maxsim_scores = score_token_docs(
        &LateInteractionScorer::MaxSimDot,
        &query_tokens,
        &doc_tokens,
    );
    
    assert_eq!(maxsim_scores.len(), 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: Edge Cases in Pipeline
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_edge_cases() {
    const DIM: usize = 32;
    
    // Empty query tokens
    let empty_query: Vec<Vec<f32>> = vec![];
    let doc = mock_token_embed("some document", DIM);
    let score = simd::maxsim_vecs(&empty_query, &doc);
    assert_eq!(score, 0.0, "Empty query should return 0");
    
    // Empty document tokens
    let query = mock_token_embed("some query", DIM);
    let empty_doc: Vec<Vec<f32>> = vec![];
    let score = simd::maxsim_vecs(&query, &empty_doc);
    assert_eq!(score, 0.0, "Empty doc should return 0");
    
    // Single token
    let single_query = vec![mock_dense_embed("word", DIM)];
    let single_doc = vec![mock_dense_embed("word", DIM)];
    let score = simd::maxsim_vecs(&single_query, &single_doc);
    assert!(score > 0.9, "Identical single tokens should have high score");
    
    // Pooling edge case: pooling still works with few tokens
    let few_tokens = vec![mock_dense_embed("one", DIM), mock_dense_embed("two", DIM)];
    let pooled = colbert::pool_tokens_adaptive(&few_tokens, 10);
    // Factor 10 on 2 tokens = target 1, so it pools down
    assert!(pooled.len() <= 2, "Should pool tokens");
    assert!(!pooled.is_empty(), "Should not produce empty result");
}

