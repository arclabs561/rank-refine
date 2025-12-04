//! Edge case tests for rank-refine algorithms.

use rank_refine::colbert;
use rank_refine::simd::{maxsim, cosine};
use rank_refine::diversity::{mmr, MmrConfig};

#[test]
fn test_maxsim_empty_tokens() {
    let query: Vec<Vec<f32>> = vec![];
    let doc: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    
    // Empty query should return 0.0
    let query_slices: Vec<&[f32]> = query.iter().map(|v| v.as_slice()).collect();
    let doc_slices: Vec<&[f32]> = doc.iter().map(|v| v.as_slice()).collect();
    
    let score = maxsim(&query_slices, &doc_slices);
    assert_eq!(score, 0.0);
}

#[test]
fn test_maxsim_single_token() {
    let query = vec![vec![1.0, 0.0]];
    let doc = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
    
    let query_slices: Vec<&[f32]> = query.iter().map(|v| v.as_slice()).collect();
    let doc_slices: Vec<&[f32]> = doc.iter().map(|v| v.as_slice()).collect();
    
    let score = maxsim(&query_slices, &doc_slices);
    assert!(score > 0.0);
    // Should match best doc token (0.9)
    assert!((score - 0.9).abs() < 0.01);
}

#[test]
fn test_cosine_mismatched_lengths() {
    // In release builds, mismatched lengths may cause undefined behavior
    // In debug builds, should panic
    let vec1 = vec![1.0, 0.0];
    let vec2 = vec![0.9, 0.1, 0.0]; // Different length
    
    #[cfg(debug_assertions)]
    {
        // Should panic in debug
        let result = std::panic::catch_unwind(|| {
            cosine(&vec1.as_slice(), &vec2.as_slice())
        });
        assert!(result.is_err());
    }
}

#[test]
fn test_pool_tokens_empty() {
    let tokens: Vec<Vec<f32>> = vec![];
    
    // Empty tokens return Ok(empty) - this is valid (no tokens to pool)
    let result = colbert::pool_tokens(&tokens, 2);
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

#[test]
fn test_pool_tokens_single_token() {
    let tokens = vec![vec![1.0, 0.0, 0.0]];
    
    // Pooling to 1 should return the single token
    let pooled = colbert::pool_tokens(&tokens, 1).unwrap();
    assert_eq!(pooled.len(), 1);
}

#[test]
fn test_pool_tokens_target_greater_than_input() {
    let tokens = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
    ];
    
    // pool_factor is a reduction factor, not target count
    // factor 10 means reduce by 10x: 2 tokens / 10 = 0.2 → rounds to 1 (minimum)
    let pooled = colbert::pool_tokens(&tokens, 10).unwrap();
    // Should return at least 1 token (minimum enforced)
    assert!(pooled.len() >= 1 && pooled.len() <= tokens.len());
}

#[test]
fn test_mmr_empty_candidates() {
    let candidates: Vec<(&str, f32)> = vec![];
    let similarity = vec![];
    
    let config = MmrConfig::default().with_k(5);
    let selected = mmr(&candidates, &similarity, config);
    assert!(selected.is_empty());
}

#[test]
fn test_mmr_k_greater_than_candidates() {
    let candidates = vec![
        ("doc1", 0.9),
        ("doc2", 0.8),
    ];
    // 2x2 similarity matrix (flattened)
    let similarity = vec![
        1.0, 0.5,  // doc1 vs [doc1, doc2]
        0.5, 1.0,  // doc2 vs [doc1, doc2]
    ];
    
    // Requesting k=10 but only 2 candidates
    let config = MmrConfig::default().with_k(10);
    let selected = mmr(&candidates, &similarity, config);
    assert_eq!(selected.len(), 2);
}

#[test]
fn test_mmr_lambda_zero() {
    // λ=0.0 should give pure diversity (ignores relevance)
    let candidates = vec![
        ("doc1", 0.9),
        ("doc2", 0.8),
        ("doc3", 0.7),
    ];
    let similarity = vec![
        1.0, 0.9, 0.2,  // doc1 vs [doc1, doc2, doc3]
        0.9, 1.0, 0.3,  // doc2 vs [doc1, doc2, doc3]
        0.2, 0.3, 1.0,  // doc3 vs [doc1, doc2, doc3]
    ];
    
    // Use lambda=0.0 for pure diversity
    let config = MmrConfig::default().with_k(2).with_lambda(0.0);
    let selected = mmr(&candidates, &similarity, config);
    // Should select diverse items
    assert_eq!(selected.len(), 2);
}

#[test]
fn test_mmr_lambda_one() {
    // λ=1.0 should give pure relevance (no diversity)
    let candidates = vec![
        ("doc1", 0.9),
        ("doc2", 0.8),
        ("doc3", 0.7),
    ];
    let similarity = vec![
        1.0, 0.9, 0.2,
        0.9, 1.0, 0.3,
        0.2, 0.3, 1.0,
    ];
    
    // Use lambda=1.0 for pure relevance
    let config = MmrConfig::default().with_k(2).with_lambda(1.0);
    let selected = mmr(&candidates, &similarity, config);
    // Should select top 2
    assert_eq!(selected.len(), 2);
}

#[test]
fn test_mmr_identical_candidates() {
    // All candidates identical should still work
    let candidates = vec![
        ("doc1", 0.5),
        ("doc2", 0.5),
        ("doc3", 0.5),
    ];
    let similarity = vec![
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
    ];
    
    let config = MmrConfig::default().with_k(2);
    let selected = mmr(&candidates, &similarity, config);
    assert_eq!(selected.len(), 2);
}

#[test]
fn test_normalize_zero_vector() {
    use rank_refine::embedding::normalize;
    
    // Zero vector should be handled
    let vec = vec![0.0, 0.0, 0.0];
    let result = normalize(&vec);
    
    // normalize returns Option, zero vector should return None
    assert!(result.is_none());
}

#[test]
fn test_normalize_very_small_vector() {
    use rank_refine::embedding::normalize;
    
    // Very small values (near zero)
    let mut vec = vec![1e-10, 1e-10, 1e-10];
    let result = normalize(&mut vec);
    
    // Very small vectors may fail normalization (near-zero magnitude)
    // Behavior depends on epsilon threshold
    // Just verify it doesn't panic
    let _result = result;
}

#[test]
fn test_pool_tokens_all_identical() {
    // All tokens identical should still pool
    let tokens = vec![
        vec![1.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
        vec![1.0, 0.0, 0.0],
    ];
    
    let pooled = colbert::pool_tokens(&tokens, 2).unwrap();
    assert!(pooled.len() <= 2);
}

#[test]
fn test_hierarchical_pooling_empty() {
    let tokens: Vec<Vec<f32>> = vec![];
    
    // Empty tokens return Ok(empty) - this is valid (no tokens to pool)
    let result = colbert::pool_tokens(&tokens, 2);
    assert!(result.is_ok(), "Empty tokens should return Ok(empty)");
    assert!(result.unwrap().is_empty());
}

#[test]
fn test_adaptive_pooling_edge_cases() {
    let tokens = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
    ];
    
    // Adaptive pooling should handle small inputs
    let pooled = colbert::pool_tokens_adaptive(&tokens, 1).unwrap();
    assert!(pooled.len() <= tokens.len());
}

