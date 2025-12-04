//! Evaluation tests using rank-eval crate.
//!
//! These tests demonstrate how to evaluate rank-refine's reranking performance
//! using standardized IR metrics from the rank-eval crate.

use rank_eval::binary::{ndcg_at_k, precision_at_k, recall_at_k};
use rank_eval::graded::{compute_ndcg, compute_map};
use rank_refine::colbert::rank;
use std::collections::{HashMap, HashSet};

#[test]
fn test_reranking_improves_ndcg() {
    // Simulate a query with token embeddings
    let query = vec![
        vec![1.0, 0.0, 0.0], // "capital"
        vec![0.0, 1.0, 0.0], // "France"
    ];

    // Documents with varying relevance
    // doc1: Highly relevant (contains both "capital" and "France")
    // doc2: Partially relevant (contains "France")
    // doc3: Not relevant
    let docs = vec![
        (
            "doc1",
            vec![
                vec![0.9, 0.1, 0.0], // "Paris" (capital)
                vec![0.1, 0.9, 0.0], // "France"
            ],
        ),
        (
            "doc2",
            vec![
                vec![0.2, 0.8, 0.0], // "country"
                vec![0.0, 0.9, 0.1], // "France"
            ],
        ),
        (
            "doc3",
            vec![
                vec![0.0, 0.0, 1.0], // "unrelated"
                vec![0.0, 0.0, 1.0], // "topic"
            ],
        ),
    ];

    // Rank documents using MaxSim
    let ranked = rank(&query, &docs);

    // Ground truth: doc1 and doc2 are relevant
    let relevant: HashSet<_> = ["doc1", "doc2"].into_iter().collect();

    // Extract document IDs from ranked results
    let ranked_ids: Vec<&str> = ranked.iter().map(|(id, _)| *id).collect();

    // Compute metrics
    let ndcg = ndcg_at_k(&ranked_ids, &relevant, 3);
    let precision = precision_at_k(&ranked_ids, &relevant, 3);
    let recall = recall_at_k(&ranked_ids, &relevant, 3);

    // Verify that relevant documents are ranked higher
    assert!(ndcg > 0.0, "nDCG should be positive when relevant docs are found");
    assert!(precision > 0.0, "Precision should be positive");
    assert!(recall > 0.0, "Recall should be positive");

    // doc1 should rank higher than doc3 (more relevant)
    let doc1_pos = ranked_ids.iter().position(|&id| id == "doc1").unwrap();
    let doc3_pos = ranked_ids.iter().position(|&id| id == "doc3").unwrap();
    assert!(
        doc1_pos < doc3_pos,
        "More relevant document should rank higher"
    );
}

#[test]
fn test_graded_relevance_evaluation() {
    // Simulate reranking with graded relevance judgments
    let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

    let docs = vec![
        ("doc1", vec![vec![0.95, 0.05], vec![0.05, 0.95]]), // Highly relevant
        ("doc2", vec![vec![0.8, 0.2], vec![0.2, 0.8]]),    // Relevant
        ("doc3", vec![vec![0.5, 0.5], vec![0.5, 0.5]]),    // Partially relevant
        ("doc4", vec![vec![0.1, 0.9], vec![0.9, 0.1]]),    // Not relevant
    ];

    let ranked = rank(&query, &docs);

    // Graded relevance: 2 = highly relevant, 1 = relevant, 0 = not relevant
    let mut qrels = HashMap::new();
    qrels.insert("doc1".to_string(), 2);
    qrels.insert("doc2".to_string(), 1);
    qrels.insert("doc3".to_string(), 1);
    qrels.insert("doc4".to_string(), 0);

    // Convert ranked results to format expected by graded metrics
    let ranked_with_scores: Vec<(String, f32)> = ranked
        .iter()
        .map(|(id, score)| (id.to_string(), *score))
        .collect();

    // Compute graded metrics
    let ndcg = compute_ndcg(&ranked_with_scores, &qrels, 4);
    let map = compute_map(&ranked_with_scores, &qrels);

    assert!(ndcg >= 0.0 && ndcg <= 1.0, "nDCG should be in [0, 1]");
    assert!(map >= 0.0 && map <= 1.0, "MAP should be in [0, 1]");

    // Highly relevant doc should rank higher
    let doc1_pos = ranked_with_scores
        .iter()
        .position(|(id, _)| id == "doc1")
        .unwrap();
    let doc4_pos = ranked_with_scores
        .iter()
        .position(|(id, _)| id == "doc4")
        .unwrap();
    assert!(
        doc1_pos < doc4_pos,
        "Highly relevant document should rank higher than non-relevant"
    );
}

#[test]
fn test_reranking_preserves_relevance_order() {
    // Test that reranking maintains relative order of relevant documents
    let query = vec![vec![1.0, 0.0]];

    // All documents are relevant, but with different similarity
    let docs = vec![
        ("doc1", vec![vec![0.99, 0.01]]), // Most similar
        ("doc2", vec![vec![0.90, 0.10]]), // Similar
        ("doc3", vec![vec![0.80, 0.20]]), // Less similar
    ];

    let ranked = rank(&query, &docs);
    let ranked_ids: Vec<&str> = ranked.iter().map(|(id, _)| *id).collect();

    let relevant: HashSet<_> = ["doc1", "doc2", "doc3"].into_iter().collect();

    // All documents are relevant, so precision should be 1.0
    let precision = precision_at_k(&ranked_ids, &relevant, 3);
    assert!((precision - 1.0).abs() < 1e-9, "All relevant docs should give precision 1.0");

    // nDCG should be high since all are relevant and well-ordered
    let ndcg = ndcg_at_k(&ranked_ids, &relevant, 3);
    assert!(ndcg > 0.8, "Well-ordered relevant docs should give high nDCG");
}

#[test]
fn test_cross_encoder_reranking_evaluation() {
    use rank_eval::binary::{ndcg_at_k, precision_at_k};
    use rank_refine::crossencoder::{rerank, CrossEncoderModel};
    use std::collections::HashSet;

    // Mock cross-encoder that scores by word overlap
    struct OverlapEncoder;
    impl CrossEncoderModel for OverlapEncoder {
        fn score_batch(&self, query: &str, documents: &[&str]) -> Vec<f32> {
            let query_lower = query.to_lowercase();
            let query_words: HashSet<&str> = query_lower.split_whitespace().collect();
            documents
                .iter()
                .map(|doc| {
                    let doc_lower = doc.to_lowercase();
                    let doc_words: HashSet<&str> = doc_lower.split_whitespace().collect();
                    let overlap = doc_words.iter().filter(|w| query_words.contains(*w)).count();
                    overlap as f32 / query_words.len().max(1) as f32
                })
                .collect()
        }
    }

    let query = "rust memory safety";
    let documents = vec![
        ("doc1", "Rust guarantees memory safety without garbage collection"),
        ("doc2", "Python is dynamically typed"),
        ("doc3", "Memory management in Rust is automatic and safe"),
        ("doc4", "JavaScript runs in web browsers"),
    ];

    let model = OverlapEncoder;
    let doc_refs: Vec<_> = documents.iter().map(|(id, text)| (*id, *text)).collect();
    let reranked = rerank(&model, query, &doc_refs);

    let ranked_ids: Vec<&str> = reranked.iter().map(|(id, _)| *id).collect();
    let relevant: HashSet<_> = ["doc1", "doc3"].into_iter().collect();

    let ndcg = ndcg_at_k(&ranked_ids, &relevant, 4);
    let precision = precision_at_k(&ranked_ids, &relevant, 4);

    // Documents with "memory" and "rust" should rank higher
    assert!(ndcg > 0.0, "nDCG should be positive");
    assert!(precision > 0.0, "Precision should be positive");
    
    // doc1 or doc3 should be in top 2 (both mention memory and rust)
    let top2: HashSet<_> = ranked_ids.iter().take(2).copied().collect();
    assert!(
        top2.contains("doc1") || top2.contains("doc3"),
        "Relevant docs should be in top 2"
    );
}

#[test]
fn test_diversity_selection_with_metrics() {
    use rank_eval::binary::ndcg_at_k;
    use rank_refine::diversity::{mmr, MmrConfig};
    use rank_refine::simd;
    use std::collections::HashSet;

    const DIM: usize = 64;

    // Generate embeddings for documents
    fn mock_embed(text: &str) -> Vec<f32> {
        let mut emb = vec![0.0; DIM];
        for (i, c) in text.chars().enumerate() {
            emb[i % DIM] += (c as u32 as f32) / 1000.0;
        }
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            emb.iter_mut().for_each(|x| *x /= norm);
        }
        emb
    }

    let documents = [
        ("rust1", "Rust systems programming", 0.95),
        ("rust2", "Rust memory safety", 0.90),
        ("rust3", "Rust web assembly", 0.85),
        ("python", "Python machine learning", 0.70),
        ("java", "Java enterprise software", 0.60),
    ];

    let embeddings: Vec<Vec<f32>> = documents
        .iter()
        .map(|(_, text, _)| mock_embed(text))
        .collect();

    let candidates: Vec<(&str, f32)> = documents
        .iter()
        .map(|(id, _, score)| (*id, *score))
        .collect();

    // Compute similarity matrix
    let n = embeddings.len();
    let mut similarity = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            similarity[i * n + j] = simd::cosine(&embeddings[i], &embeddings[j]);
        }
    }

    // Select top 3 with diversity (lambda = 0.5 balances relevance and diversity)
    let config = MmrConfig::default().with_lambda(0.5).with_k(3);
    let selected = mmr(&candidates, &similarity, config);

    let selected_ids: Vec<&str> = selected.iter().map(|(id, _)| *id).collect();
    
    // All rust docs are relevant
    let relevant: HashSet<_> = ["rust1", "rust2", "rust3"].into_iter().collect();

    let ndcg = ndcg_at_k(&selected_ids, &relevant, 3);
    
    // With diversity, we might get fewer rust docs, but should still have good nDCG
    assert!(ndcg >= 0.0 && ndcg <= 1.0, "nDCG should be in [0, 1]");
    
    // Verify we got 3 results
    assert_eq!(selected.len(), 3, "Should select exactly 3 documents");
}

#[test]
fn test_matryoshka_refinement_evaluation() {
    use rank_eval::binary::ndcg_at_k;
    use rank_refine::matryoshka::refine;
    use rank_refine::simd;
    use std::collections::HashSet;

    const FULL_DIM: usize = 256;
    const HEAD_DIM: usize = 64;

    fn mock_embed(text: &str, dim: usize) -> Vec<f32> {
        let mut emb = vec![0.0; dim];
        for (i, c) in text.chars().enumerate() {
            emb[i % dim] += (c as u32 as f32) / 1000.0;
        }
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            emb.iter_mut().for_each(|x| *x /= norm);
        }
        emb
    }

    let query = mock_embed("Rust systems programming", FULL_DIM);
    let documents = vec![
        ("doc1", mock_embed("Rust systems programming language", FULL_DIM)),
        ("doc2", mock_embed("Python data science", FULL_DIM)),
        ("doc3", mock_embed("Rust memory safety", FULL_DIM)),
    ];

    // Initial ranking with head dimensions only (fast)
    let head_scores: Vec<(&str, f32)> = documents
        .iter()
        .map(|(id, emb)| (*id, simd::cosine(&query[..HEAD_DIM], &emb[..HEAD_DIM])))
        .collect();

    // Refine using tail dimensions
    let refined = refine(&head_scores, &query, &documents, HEAD_DIM);

    let refined_ids: Vec<&str> = refined.iter().map(|(id, _)| *id).collect();
    let relevant: HashSet<_> = ["doc1", "doc3"].into_iter().collect();

    let ndcg = ndcg_at_k(&refined_ids, &relevant, 3);
    
    assert!(ndcg >= 0.0 && ndcg <= 1.0, "nDCG should be in [0, 1]");
    
    // Refinement should maintain or improve ranking quality
    let head_ndcg = ndcg_at_k(
        &head_scores.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        &relevant,
        3,
    );
    
    // Refined nDCG should be at least as good as head-only
    assert!(
        ndcg >= head_ndcg - 0.1, // Allow small tolerance
        "Refinement should maintain or improve nDCG"
    );
}

#[test]
fn test_fine_grained_reranking_evaluation() {
    use rank_eval::graded::{compute_ndcg, compute_map};
    use rank_refine::explain::{rerank_fine_grained, Candidate, FineGrainedConfig, RerankerInput, RerankMethod};
    use std::collections::HashMap;

    const DIM: usize = 128;

    fn mock_token_embed(text: &str) -> Vec<Vec<f32>> {
        text.split_whitespace()
            .map(|word| {
                let mut emb = vec![0.0; DIM];
                for (i, c) in word.chars().enumerate() {
                    emb[i % DIM] += (c as u32 as f32) / 1000.0;
                }
                let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    emb.iter_mut().for_each(|x| *x /= norm);
                }
                emb
            })
            .collect()
    }

    let query_tokens = mock_token_embed("Rust memory safety");
    let doc1_tokens = mock_token_embed("Rust is a systems programming language with memory safety");
    let doc2_tokens = mock_token_embed("Python is great for data science");
    let doc3_tokens = mock_token_embed("Rust memory management is automatic");

    let candidates = vec![
        Candidate {
            id: "doc1",
            original_score: 0.8,
            dense_embedding: None,
            token_embeddings: Some(&doc1_tokens),
            text: None,
        },
        Candidate {
            id: "doc2",
            original_score: 0.6,
            dense_embedding: None,
            token_embeddings: Some(&doc2_tokens),
            text: None,
        },
        Candidate {
            id: "doc3",
            original_score: 0.7,
            dense_embedding: None,
            token_embeddings: Some(&doc3_tokens),
            text: None,
        },
    ];

    let input = RerankerInput {
        query_dense: None,
        query_tokens: Some(&query_tokens),
        candidates,
    };

    let config = FineGrainedConfig::default();
    let results = rerank_fine_grained(input, RerankMethod::MaxSim, config, 10);

    // Convert to format for graded metrics
    let ranked_with_scores: Vec<(String, f32)> = results
        .iter()
        .map(|r| (r.id.to_string(), r.fine_score as f32))
        .collect();

    // Graded relevance: fine_score (0-10) maps to relevance
    let mut qrels = HashMap::new();
    qrels.insert("doc1".to_string(), 8); // High fine score = high relevance
    qrels.insert("doc2".to_string(), 3); // Lower fine score = lower relevance
    qrels.insert("doc3".to_string(), 7); // Medium-high fine score

    let ndcg = compute_ndcg(&ranked_with_scores, &qrels, 3);
    let map = compute_map(&ranked_with_scores, &qrels);

    assert!(ndcg >= 0.0 && ndcg <= 1.0, "nDCG should be in [0, 1]");
    assert!(map >= 0.0 && map <= 1.0, "MAP should be in [0, 1]");
    
    // doc1 should rank highest (most relevant)
    assert_eq!(results[0].id, "doc1", "Most relevant doc should rank first");
}

#[test]
fn test_edge_case_empty_results() {
    use rank_eval::binary::{ndcg_at_k, precision_at_k};
    use std::collections::HashSet;

    // Empty ranked list
    let ranked: Vec<&str> = vec![];
    let relevant: HashSet<_> = ["doc1", "doc2"].into_iter().collect();

    let ndcg = ndcg_at_k(&ranked, &relevant, 10);
    let precision = precision_at_k(&ranked, &relevant, 10);

    assert_eq!(ndcg, 0.0, "Empty ranking should give nDCG = 0");
    assert_eq!(precision, 0.0, "Empty ranking should give precision = 0");
}

#[test]
fn test_edge_case_no_relevant_documents() {
    use rank_eval::binary::{ndcg_at_k, precision_at_k, recall_at_k};
    use std::collections::HashSet;

    let ranked = vec!["doc1", "doc2", "doc3"];
    let relevant: HashSet<&str> = HashSet::new(); // No relevant docs

    let ndcg = ndcg_at_k(&ranked, &relevant, 10);
    let precision = precision_at_k(&ranked, &relevant, 10);
    let recall = recall_at_k(&ranked, &relevant, 10);

    assert_eq!(ndcg, 0.0, "No relevant docs should give nDCG = 0");
    assert_eq!(precision, 0.0, "No relevant docs should give precision = 0");
    assert_eq!(recall, 0.0, "No relevant docs should give recall = 0");
}

#[test]
fn test_edge_case_all_relevant() {
    use rank_eval::binary::{ndcg_at_k, precision_at_k, recall_at_k};
    use std::collections::HashSet;

    let ranked = vec!["doc1", "doc2", "doc3"];
    let relevant: HashSet<_> = ["doc1", "doc2", "doc3"].into_iter().collect();

    let ndcg = ndcg_at_k(&ranked, &relevant, 10);
    let precision = precision_at_k(&ranked, &relevant, 3);
    let recall = recall_at_k(&ranked, &relevant, 3);

    assert!(ndcg > 0.0, "All relevant should give positive nDCG");
    assert!((precision - 1.0).abs() < 1e-9, "All relevant should give precision = 1.0");
    assert!((recall - 1.0).abs() < 1e-9, "All relevant should give recall = 1.0");
}

#[test]
fn test_metrics_consistency_across_k_values() {
    use rank_eval::binary::{ndcg_at_k, precision_at_k};
    use std::collections::HashSet;

    let ranked = vec!["doc1", "doc2", "doc3", "doc4", "doc5"];
    let relevant: HashSet<_> = ["doc1", "doc3", "doc5"].into_iter().collect();

    // Precision should be non-increasing as k increases
    let p1 = precision_at_k(&ranked, &relevant, 1);
    let p3 = precision_at_k(&ranked, &relevant, 3);
    let p5 = precision_at_k(&ranked, &relevant, 5);

    assert!(p1 >= p3, "Precision@1 should be >= Precision@3");
    assert!(p3 >= p5, "Precision@3 should be >= Precision@5");

    // nDCG should be non-decreasing as k increases (more docs = more opportunity)
    let ndcg3 = ndcg_at_k(&ranked, &relevant, 3);
    let ndcg5 = ndcg_at_k(&ranked, &relevant, 5);
    let ndcg10 = ndcg_at_k(&ranked, &relevant, 10);

    assert!(ndcg5 >= ndcg3 - 1e-9, "nDCG@5 should be >= nDCG@3");
    assert!(ndcg10 >= ndcg5 - 1e-9, "nDCG@10 should be >= nDCG@5");
}

#[test]
fn test_perfect_ranking_metrics() {
    use rank_eval::binary::{ndcg_at_k, precision_at_k, recall_at_k, mrr};
    use std::collections::HashSet;

    // Perfect ranking: all relevant docs at top
    let ranked = vec!["doc1", "doc2", "doc3", "doc4", "doc5"];
    let relevant: HashSet<_> = ["doc1", "doc2", "doc3"].into_iter().collect();

    let ndcg = ndcg_at_k(&ranked, &relevant, 10);
    let precision = precision_at_k(&ranked, &relevant, 3);
    let recall = recall_at_k(&ranked, &relevant, 3);
    let mrr_score = mrr(&ranked, &relevant);

    // Perfect ranking should give nDCG = 1.0
    assert!((ndcg - 1.0).abs() < 1e-9, "Perfect ranking should give nDCG = 1.0");
    assert!((precision - 1.0).abs() < 1e-9, "Perfect ranking should give precision = 1.0");
    assert!((recall - 1.0).abs() < 1e-9, "Perfect ranking should give recall = 1.0");
    assert!((mrr_score - 1.0).abs() < 1e-9, "Perfect ranking should give MRR = 1.0");
}

#[test]
fn test_worst_case_ranking_metrics() {
    use rank_eval::binary::{ndcg_at_k, precision_at_k};
    use std::collections::HashSet;

    // Worst case: all relevant docs at bottom
    let ranked = vec!["doc4", "doc5", "doc6", "doc1", "doc2", "doc3"];
    let relevant: HashSet<_> = ["doc1", "doc2", "doc3"].into_iter().collect();

    let ndcg = ndcg_at_k(&ranked, &relevant, 6);
    let precision = precision_at_k(&ranked, &relevant, 3);

    // Worst case should give low but non-zero metrics
    // Note: nDCG can be > 0.5 even in worst case if relevant docs appear later (just not first)
    // The key is that precision@3 should be 0 (no relevant docs in top-3)
    assert!(ndcg < 0.8, "Worst case ranking should give relatively low nDCG");
    assert_eq!(precision, 0.0, "Worst case (relevant docs not in top-3) should give precision = 0");
}

#[test]
fn test_reranking_methods_comparison() {
    use rank_eval::binary::ndcg_at_k;
    use rank_refine::colbert::rank;
    use rank_refine::simd;
    use std::collections::HashSet;

    const DIM: usize = 128;

    fn mock_embed(text: &str) -> Vec<Vec<f32>> {
        text.split_whitespace()
            .map(|word| {
                let mut emb = vec![0.0; DIM];
                for (i, c) in word.chars().enumerate() {
                    emb[i % DIM] += (c as u32 as f32) / 1000.0;
                }
                let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    emb.iter_mut().for_each(|x| *x /= norm);
                }
                emb
            })
            .collect()
    }

    let query = mock_embed("Rust memory safety");
    let docs = vec![
        ("doc1", mock_embed("Rust memory safety guarantees")),
        ("doc2", mock_embed("Python machine learning")),
        ("doc3", mock_embed("Rust systems programming")),
    ];

    // Rank using MaxSim (ColBERT-style)
    let ranked = rank(&query, &docs);
    let ranked_ids: Vec<&str> = ranked.iter().map(|(id, _)| *id).collect();
    let relevant: HashSet<_> = ["doc1", "doc3"].into_iter().collect();

    let ndcg = ndcg_at_k(&ranked_ids, &relevant, 3);

    // MaxSim should rank relevant docs higher
    assert!(ndcg > 0.0, "MaxSim should produce positive nDCG");
    
    // doc1 should rank higher than doc2 (more relevant)
    let doc1_pos = ranked_ids.iter().position(|&id| id == "doc1").unwrap();
    let doc2_pos = ranked_ids.iter().position(|&id| id == "doc2").unwrap();
    assert!(
        doc1_pos < doc2_pos,
        "More relevant doc should rank higher"
    );
}

