//! E2E test: rank-refine + rank-eval integration.

use rank_refine::colbert;
use rank_eval::binary::ndcg_at_k;
use rank_eval::graded::compute_ndcg;
use std::collections::{HashMap, HashSet};

fn main() {
    println!("Testing rank-refine + rank-eval integration...");
    
    // Simulate query and document token embeddings
    let query_tokens = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
    ];
    
    let doc1_tokens = vec![
        vec![0.9, 0.1, 0.0],
        vec![0.1, 0.9, 0.0],
    ];
    
    let doc2_tokens = vec![
        vec![0.95, 0.05, 0.0],
        vec![0.05, 0.95, 0.0],
    ];
    
    let doc3_tokens = vec![
        vec![0.8, 0.2, 0.0],
        vec![0.2, 0.8, 0.0],
    ];
    
    // Rank documents with ColBERT
    let docs = vec![
        ("doc1".to_string(), doc1_tokens),
        ("doc2".to_string(), doc2_tokens),
        ("doc3".to_string(), doc3_tokens),
    ];
    
    let refined = colbert::rank(&query_tokens, &docs);
    assert!(!refined.is_empty());
    println!("✅ Refined {} documents", refined.len());
    
    // Extract ranked list for evaluation
    let ranked: Vec<String> = refined.iter().map(|(id, _)| id.clone()).collect();
    
    // Binary relevance
    let relevant: HashSet<String> = ["doc1", "doc2"].iter()
        .map(|s| s.to_string())
        .collect();
    
    // Compute binary metrics
    let ndcg = ndcg_at_k(&ranked, &relevant, 10);
    assert!(ndcg >= 0.0 && ndcg <= 1.0);
    println!("✅ Binary nDCG@10: {:.4}", ndcg);
    
    // Graded relevance
    let mut qrels = HashMap::new();
    qrels.insert("doc1".to_string(), 2);
    qrels.insert("doc2".to_string(), 1);
    qrels.insert("doc3".to_string(), 0);
    
    let refined_graded: Vec<(String, f32)> = refined.iter()
        .map(|(id, score)| (id.clone(), *score))
        .collect();
    
    let graded_ndcg = compute_ndcg(&refined_graded, &qrels, 10);
    assert!(graded_ndcg >= 0.0 && graded_ndcg <= 1.0);
    println!("✅ Graded nDCG@10: {:.4}", graded_ndcg);
    
    println!("\n✅ All refine-eval integration tests passed!");
}

