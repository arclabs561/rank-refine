//! E2E test: Basic rank-refine usage as a published crate.

use rank_refine::colbert;
use rank_refine::simd::{maxsim, cosine};

fn main() {
    println!("Testing rank-refine as published crate...");
    
    // Simulate query and document token embeddings
    let query_tokens = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    
    let doc1_tokens = vec![
        vec![0.9, 0.1, 0.0],
        vec![0.1, 0.9, 0.0],
        vec![0.0, 0.1, 0.9],
    ];
    
    let doc2_tokens = vec![
        vec![0.8, 0.2, 0.0],
        vec![0.2, 0.8, 0.0],
        vec![0.0, 0.2, 0.8],
    ];
    
    // Test MaxSim scoring - convert to slices
    let query_slices: Vec<&[f32]> = query_tokens.iter().map(|v| v.as_slice()).collect();
    let doc1_slices: Vec<&[f32]> = doc1_tokens.iter().map(|v| v.as_slice()).collect();
    let doc2_slices: Vec<&[f32]> = doc2_tokens.iter().map(|v| v.as_slice()).collect();
    
    let score1 = maxsim(&query_slices, &doc1_slices);
    let score2 = maxsim(&query_slices, &doc2_slices);
    
    assert!(score1 > 0.0 && score1.is_finite());
    assert!(score2 > 0.0 && score2.is_finite());
    println!("✅ MaxSim scores: doc1={:.4}, doc2={:.4}", score1, score2);
    
    // Test ColBERT ranking
    let docs = vec![
        ("doc1".to_string(), doc1_tokens.clone()),
        ("doc2".to_string(), doc2_tokens.clone()),
    ];
    
    let ranked = colbert::rank(&query_tokens, &docs);
    assert!(!ranked.is_empty());
    assert_eq!(ranked.len(), 2);
    println!("✅ ColBERT ranking: {} results", ranked.len());
    
    // Test token pooling
    let pooled = colbert::pool_tokens(&doc1_tokens, 2).unwrap();
    assert!(!pooled.is_empty());
    assert!(pooled.len() <= doc1_tokens.len());
    println!("✅ Token pooling: {} -> {} tokens", doc1_tokens.len(), pooled.len());
    
    // Test cosine similarity
    let vec1 = vec![1.0, 0.0, 0.0];
    let vec2 = vec![0.9, 0.1, 0.0];
    let cos_sim = cosine(&vec1.as_slice(), &vec2.as_slice());
    
    assert!(cos_sim >= -1.0 && cos_sim <= 1.0);
    println!("✅ Cosine similarity: {:.4}", cos_sim);
    
    println!("\n✅ All rank-refine basic tests passed!");
}

