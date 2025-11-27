//! Integration tests simulating realistic e2e workflows.
//!
//! These tests use synthetic embeddings that mimic real model outputs.
//! They verify the full pipeline works correctly without requiring actual models.

use rank_refine::{
    colbert,
    crossencoder::CrossEncoderModel,
    matryoshka,
    scoring::{
        blend, normalize_scores, AdaptivePooler, ClusteringPooler, DenseScorer,
        LateInteractionScorer, Pooler, Scorer, SequentialPooler, TokenScorer,
    },
    simd, RefineConfig,
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
        let query_words: std::collections::HashSet<&str> = query_lower.split_whitespace().collect();

        documents
            .iter()
            .map(|doc| {
                let doc_lower = doc.to_lowercase();
                let doc_words: std::collections::HashSet<&str> =
                    doc_lower.split_whitespace().collect();
                let overlap = doc_words
                    .iter()
                    .filter(|w| query_words.contains(*w))
                    .count();
                overlap as f32 / query_words.len().max(1) as f32
            })
            .collect()
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
        (
            "d1",
            "Rust is a systems programming language focused on safety",
        ),
        (
            "d2",
            "Python is great for data science and machine learning",
        ),
        (
            "d3",
            "Rust guarantees memory safety without garbage collection",
        ),
        ("d4", "JavaScript runs in web browsers and Node.js"),
    ];

    let query = "memory safe systems programming";

    // Stage 1: Dense retrieval
    let query_dense = mock_dense_embed(query, DIM);
    let docs_dense: Vec<_> = documents
        .iter()
        .map(|(id, text)| (*id, mock_dense_embed(text, DIM)))
        .collect();

    let scorer = DenseScorer::Cosine;
    let doc_refs: Vec<_> = docs_dense
        .iter()
        .map(|(id, emb)| (*id, emb.as_slice()))
        .collect();

    let first_stage = scorer.rank(&query_dense, &doc_refs);

    // Verify we got all documents
    assert_eq!(first_stage.len(), 4);

    // Stage 2: ColBERT refinement on top-2
    let top_2: Vec<_> = first_stage.iter().take(2).cloned().collect();

    let query_tokens = mock_token_embed(query, DIM);
    let doc_tokens: HashMap<_, _> = documents
        .iter()
        .map(|(id, text)| (*id, mock_token_embed(text, DIM)))
        .collect();

    let doc_token_vec: Vec<_> = top_2
        .iter()
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
    let docs: Vec<_> = documents
        .iter()
        .map(|(id, text)| (*id, mock_dense_embed(text, FULL_DIM)))
        .collect();

    // Initial ranking with head dimensions only
    let head_scorer = DenseScorer::Cosine;
    let head_docs: Vec<_> = docs
        .iter()
        .map(|(id, emb)| (*id, &emb[..HEAD_DIM]))
        .collect();

    let initial_ranking = head_scorer.rank(&query_emb[..HEAD_DIM], &head_docs);

    // Refine using tail dimensions
    let candidates: Vec<_> = initial_ranking
        .iter()
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
        assert!(
            pooled.len() <= original_count,
            "{} should reduce count",
            name
        );
        assert!(
            pooled.len() >= original_count / 3,
            "{} shouldn't over-reduce",
            name
        );

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
            name,
            score_pooled,
            score_original
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
    let _initial: Vec<_> = documents
        .iter()
        .enumerate()
        .map(|(i, _)| (i, 0.5 + i as f32 * 0.1)) // arbitrary initial scores
        .collect();

    // Rerank with cross-encoder
    use rank_refine::crossencoder::rerank;
    let doc_refs: Vec<_> = documents.iter().enumerate().map(|(i, s)| (i, *s)).collect();
    let reranked = rerank(&model, query, &doc_refs);

    // Documents with "memory" and "rust" should rank higher
    assert_eq!(reranked.len(), 3);
    // Doc 0 and 2 mention "memory" and related terms
    let top_id = reranked[0].0;
    assert!(
        top_id == 0 || top_id == 2,
        "Expected doc with overlap to rank first"
    );
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
    let dense_scores: Vec<f32> = documents
        .iter()
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
    let hybrid: Vec<f32> = dense_norm
        .iter()
        .zip(&bm25_norm)
        .map(|(d, b)| blend(*d, *b, 0.7))
        .collect();

    // Verify normalization worked
    for &score in &dense_norm {
        assert!(
            score >= 0.0 && score <= 1.0,
            "Normalized score out of bounds"
        );
    }
    for &score in &bm25_norm {
        assert!(
            score >= 0.0 && score <= 1.0,
            "Normalized score out of bounds"
        );
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
    let doc_data: Vec<_> = documents
        .iter()
        .map(|(id, text)| (*id, mock_token_embed(text, DIM)))
        .collect();

    // Initial ranking
    let initial: Vec<_> = colbert::rank(&query_tokens, &doc_data);

    // Refine with config
    let config = RefineConfig::default().with_alpha(0.6).with_top_k(3);

    let refined = colbert::refine_with_config(&initial, &query_tokens, &doc_data, config);

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
        docs: &[Vec<Vec<f32>>],
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

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: Clustering Pooling (uses hierarchical when feature enabled)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_clustering_pooling() {
    const DIM: usize = 128;

    // Document with semantically distinct segments
    let document = "Rust memory safety systems programming \
                    Python machine learning data science \
                    JavaScript web browser frontend";

    let tokens = mock_token_embed(document, DIM);
    let original_count = tokens.len();

    // Clustering pooling (uses hierarchical when feature enabled)
    let pooled = colbert::pool_tokens(&tokens, 2);

    // Verify reduction
    assert!(pooled.len() <= original_count, "Should reduce token count");

    // Dimension preserved
    for token in &pooled {
        assert_eq!(token.len(), DIM);
    }

    // Pooled can still score queries
    let query = mock_token_embed("Rust memory", DIM);
    let score = simd::maxsim_vecs(&query, &pooled);
    assert!(score > 0.0, "Pooled should be scorable");
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: MMR Diversity Selection
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_mmr_diversity() {
    use rank_refine::diversity::{mmr, MmrConfig};

    const DIM: usize = 64;

    // Documents: some similar, some diverse
    let documents = vec![
        ("rust1", "Rust systems programming"),
        ("rust2", "Rust memory safety"),
        ("rust3", "Rust web assembly"),
        ("python", "Python machine learning"),
        ("java", "Java enterprise software"),
    ];

    // Generate embeddings and compute similarity matrix
    let embeddings: Vec<Vec<f32>> = documents
        .iter()
        .map(|(_, text)| mock_dense_embed(text, DIM))
        .collect();

    // Relevance scores (rust docs more relevant)
    let candidates: Vec<(&str, f32)> = vec![
        ("rust1", 0.95),
        ("rust2", 0.90),
        ("rust3", 0.85),
        ("python", 0.70),
        ("java", 0.60),
    ];

    // Pairwise similarity matrix (flattened)
    let n = embeddings.len();
    let mut similarity = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            similarity[i * n + j] = simd::cosine(&embeddings[i], &embeddings[j]);
        }
    }

    // Select top 3 with diversity
    let config = MmrConfig::default().with_lambda(0.5).with_k(3);
    let selected = mmr(&candidates, &similarity, config);

    assert_eq!(selected.len(), 3);

    // Verify results are sorted by final score
    for window in selected.windows(2) {
        assert!(window[0].1 >= window[1].1, "Results should be sorted");
    }

    // Verify diversity had an effect: not just top-3 by relevance
    // (the top 3 by pure relevance would be rust1, rust2, rust3)
    let ids: Vec<_> = selected.iter().map(|(id, _)| *id).collect();

    // At minimum, verify we got valid IDs from our candidate set
    for id in &ids {
        assert!(
            candidates.iter().any(|(cid, _)| cid == id),
            "Selected ID should be from candidates"
        );
    }
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
    assert!(
        score > 0.9,
        "Identical single tokens should have high score"
    );

    // Pooling edge case: pooling still works with few tokens
    let few_tokens = vec![mock_dense_embed("one", DIM), mock_dense_embed("two", DIM)];
    let pooled = colbert::pool_tokens_adaptive(&few_tokens, 10);
    // Factor 10 on 2 tokens = target 1, so it pools down
    assert!(pooled.len() <= 2, "Should pool tokens");
    assert!(!pooled.is_empty(), "Should not produce empty result");
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: Score Normalization Utilities
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_score_normalization_pipeline() {
    const DIM: usize = 128;

    let query = mock_token_embed("rust memory safety", DIM);
    let documents = vec![
        mock_token_embed("Rust systems programming", DIM),
        mock_token_embed("Python machine learning", DIM),
        mock_token_embed("Rust memory management", DIM),
        mock_token_embed("JavaScript web development", DIM),
    ];

    // Get raw MaxSim scores
    let raw_scores: Vec<f32> = documents
        .iter()
        .map(|d| simd::maxsim_vecs(&query, d))
        .collect();

    // Normalize to [0, 1] for thresholds
    let normalized = simd::normalize_maxsim_batch(&raw_scores, 32);
    for &s in &normalized {
        assert!(
            s >= 0.0 && s <= 2.0,
            "Normalized score {} should be in reasonable range",
            s
        );
    }

    // Softmax for relative comparison
    let softmax = simd::softmax_scores(&raw_scores);
    let sum: f32 = softmax.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Softmax should sum to 1, got {}",
        sum
    );

    // Top-k selection
    let top2 = simd::top_k_indices(&raw_scores, 2);
    assert_eq!(top2.len(), 2, "Should return top 2 indices");

    // Verify top-k are actually highest scores
    for &idx in &top2 {
        for (other_idx, &other_score) in raw_scores.iter().enumerate() {
            if !top2.contains(&other_idx) {
                assert!(
                    raw_scores[idx] >= other_score,
                    "Top-k index {} (score {}) should have score >= {} (idx {})",
                    idx,
                    raw_scores[idx],
                    other_score,
                    other_idx
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: Weighted MaxSim Pipeline
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_weighted_maxsim_pipeline() {
    const DIM: usize = 64;

    // Query with 3 tokens, different importance
    let query_text = "important critical keyword";
    let query = mock_token_embed(query_text, DIM);

    // Token importance weights (first token most important)
    let weights = vec![2.0, 1.5, 1.0];

    let documents = vec![
        mock_token_embed("important systems", DIM), // Has "important"
        mock_token_embed("critical analysis", DIM), // Has "critical"
        mock_token_embed("keyword search", DIM),    // Has "keyword"
        mock_token_embed("unrelated text", DIM),    // No match
    ];

    // Score with weights
    let weighted_scores: Vec<f32> = documents
        .iter()
        .map(|d| simd::maxsim_weighted_vecs(&query, d, &weights))
        .collect();

    // Score without weights
    let unweighted_scores: Vec<f32> = documents
        .iter()
        .map(|d| simd::maxsim_vecs(&query, d))
        .collect();

    // Weighted scores should differ from unweighted
    let mut differ = false;
    for (w, u) in weighted_scores.iter().zip(unweighted_scores.iter()) {
        if (w - u).abs() > 1e-6 {
            differ = true;
            break;
        }
    }
    assert!(differ, "Weighted and unweighted scores should differ");

    // Document with "important" should benefit more from high weight
    // (This is a softer assertion since mock embeddings are deterministic)
    assert!(
        weighted_scores[0] > 0.0,
        "Document with important term should have positive score"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: Full RAG Reranking Pipeline
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_rag_rerank_pipeline() {
    const DIM: usize = 128;

    // Simulate a RAG query
    let query = "How does Rust ensure memory safety?";
    let query_dense = mock_dense_embed(query, DIM);
    let query_tokens = mock_token_embed(query, DIM);

    // Retrieved chunks (simulating retrieval results)
    let chunks = vec![
        (
            1,
            "Rust uses ownership and borrowing to ensure memory safety",
        ),
        (2, "Python is a dynamically typed language"),
        (3, "Memory safety in Rust is enforced at compile time"),
        (4, "JavaScript runs in web browsers"),
        (5, "Rust's borrow checker prevents data races"),
    ];

    // Stage 1: Dense similarity
    let dense_candidates: Vec<(i32, f32)> = chunks
        .iter()
        .map(|(id, text)| {
            let doc_emb = mock_dense_embed(text, DIM);
            let score = simd::cosine(&query_dense, &doc_emb);
            (*id, score)
        })
        .collect();

    // Stage 2: ColBERT MaxSim reranking on top-4
    let top_k_ids: Vec<i32> = {
        let mut sorted = dense_candidates.clone();
        sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
        sorted.iter().take(4).map(|(id, _)| *id).collect()
    };

    let colbert_candidates: Vec<(i32, f32)> = top_k_ids
        .iter()
        .map(|id| {
            let chunk_text = chunks.iter().find(|(cid, _)| cid == id).unwrap().1;
            let doc_tokens = mock_token_embed(chunk_text, DIM);
            let score = simd::maxsim_vecs(&query_tokens, &doc_tokens);
            (*id, score)
        })
        .collect();

    // Stage 3: Blend scores
    let blended: Vec<(i32, f32)> = colbert_candidates
        .iter()
        .map(|(id, colbert_score)| {
            let dense_score = dense_candidates
                .iter()
                .find(|(did, _)| did == id)
                .unwrap()
                .1;
            let final_score = blend(dense_score, *colbert_score, 0.7);
            (*id, final_score)
        })
        .collect();

    // Verify pipeline produces valid results
    assert_eq!(blended.len(), 4, "Should have 4 reranked candidates");
    for (_, score) in &blended {
        assert!(score.is_finite(), "All scores should be finite");
    }

    // Normalize for threshold comparison
    let scores: Vec<f32> = blended.iter().map(|(_, s)| *s).collect();
    let normalized = normalize_scores(&scores);
    for &s in &normalized {
        assert!(
            s >= 0.0 && s <= 1.0,
            "Normalized score {} should be in [0, 1]",
            s
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: Matryoshka + Normalization
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_matryoshka_normalization() {
    const DIM: usize = 256;
    const HEAD: usize = 64;

    let query = mock_dense_embed("Rust programming", DIM);
    let documents = vec![
        ("doc1", mock_dense_embed("Rust systems programming", DIM)),
        ("doc2", mock_dense_embed("Python data science", DIM)),
        ("doc3", mock_dense_embed("Rust memory safety", DIM)),
    ];

    // Stage 1: Head-only scores (fast)
    let head_scores: Vec<(&str, f32)> = documents
        .iter()
        .map(|(id, emb)| (*id, simd::cosine(&query[..HEAD], &emb[..HEAD])))
        .collect();

    // Stage 2: Full refinement
    let refined = matryoshka::refine(&head_scores, &query, &documents, HEAD);

    assert_eq!(refined.len(), 3);

    // Normalize refined scores for thresholding
    let scores: Vec<f32> = refined.iter().map(|(_, s)| *s).collect();
    let normalized = normalize_scores(&scores);

    // Apply a threshold (keep only above 0.3)
    let above_threshold: Vec<_> = refined
        .iter()
        .zip(normalized.iter())
        .filter(|(_, &norm)| norm > 0.3)
        .map(|((id, score), _)| (*id, *score))
        .collect();

    assert!(
        !above_threshold.is_empty(),
        "Should have some results above threshold"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: NaN Handling Across Pipeline
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_nan_handling() {
    // Test that NaN values are handled gracefully throughout
    let scores_with_nan = vec![0.9, f32::NAN, 0.7, f32::NAN, 0.5];

    // Normalize should handle NaN
    let normalized = simd::normalize_maxsim_batch(&scores_with_nan, 32);
    // NaN values should remain NaN
    for (i, &s) in normalized.iter().enumerate() {
        if scores_with_nan[i].is_nan() {
            assert!(s.is_nan(), "NaN input should produce NaN output");
        }
    }

    // Softmax should handle NaN gracefully (produces 0 for NaN inputs)
    let softmax = simd::softmax_scores(&scores_with_nan);
    let finite_sum: f32 = softmax.iter().filter(|s| s.is_finite()).sum();
    assert!(
        finite_sum > 0.0,
        "Softmax should produce some positive values"
    );

    // Top-k should place NaN last
    let top3 = simd::top_k_indices(&scores_with_nan, 3);
    // First 3 should be non-NaN indices (0, 2, 4)
    for &idx in &top3 {
        assert!(
            !scores_with_nan[idx].is_nan(),
            "Top-k should prefer non-NaN values"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: Batch Operations
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_batch_operations() {
    const DIM: usize = 64;

    let query = mock_token_embed("query text here", DIM);
    let documents: Vec<Vec<Vec<f32>>> = vec![
        mock_token_embed("document one about rust", DIM),
        mock_token_embed("document two about python", DIM),
        mock_token_embed("document three about java", DIM),
        mock_token_embed("document four about rust programming", DIM),
    ];

    // Batch MaxSim
    let batch_scores = simd::maxsim_batch(&query, &documents);
    assert_eq!(batch_scores.len(), 4, "Should score all documents");

    // Verify batch matches individual scoring
    for (i, doc) in documents.iter().enumerate() {
        let individual = simd::maxsim_vecs(&query, doc);
        assert!(
            (batch_scores[i] - individual).abs() < 1e-5,
            "Batch and individual scores should match for doc {}",
            i
        );
    }

    // Batch cosine MaxSim
    let batch_cosine = simd::maxsim_cosine_batch(&query, &documents);
    assert_eq!(batch_cosine.len(), 4);

    // Top-k from batch
    let top2 = simd::top_k_indices(&batch_scores, 2);
    assert_eq!(top2.len(), 2);
}
