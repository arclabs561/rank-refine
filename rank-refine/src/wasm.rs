//! WebAssembly bindings for rank-refine.
//!
//! This module provides JavaScript-compatible bindings for rank-refine algorithms.
//! It's only compiled when the `wasm` feature is enabled.
//!
//! # Usage Example
//!
//! ```javascript
//! import init, { cosine, maxsim, pool_tokens, mmr, dpp, refine } from '@arclabs561/rank-refine';
//!
//! await init();
//!
//! // Cosine similarity
//! const score = cosine([1.0, 0.0], [0.707, 0.707]);
//!
//! // MaxSim (ColBERT)
//! const query = [[1.0, 0.0], [0.0, 1.0]];
//! const doc = [[0.9, 0.1], [0.1, 0.9]];
//! const maxsim_score = maxsim(query, doc);
//!
//! // Token pooling
//! const pooled = pool_tokens(doc, 2); // Factor 2 pooling
//!
//! // MMR diversity
//! const candidates = [["d1", 0.9], ["d2", 0.8], ["d3", 0.7]];
//! const similarity = [1.0, 0.5, 0.3, 0.5, 1.0, 0.4, 0.3, 0.4, 1.0]; // Flattened 3x3 matrix
//! const diverse = mmr(candidates, similarity, 10, 0.5); // k=10, lambda=0.5
//!
//! // Matryoshka refinement
//! const refined = refine(candidates, [0.1, 0.2, ...], docs, 64); // head_dims=64
//! ```
//!
//! # Input Format
//!
//! - **Vectors**: Arrays of numbers `[f32, f32, ...]`
//! - **Token embeddings**: Arrays of vectors `[[f32, ...], [f32, ...], ...]`
//! - **Candidates**: Arrays of `[id, score]` pairs where `id` is a string
//! - **Similarity matrices**: Flattened row-major arrays `[f32, ...]` (n×n for n candidates)

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
use crate::{
    colbert, diversity::{self, DppConfig, MmrConfig}, matryoshka, simd, RefineConfig,
};

/// Helper to convert JS array of numbers to Vec<f32>
#[cfg(feature = "wasm")]
fn js_to_vec_f32(js: &JsValue) -> Result<Vec<f32>, JsValue> {
    use wasm_bindgen::JsCast;

    let array = js
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array"))?;

    let mut vec = Vec::with_capacity(array.length() as usize);
    for (idx, item) in array.iter().enumerate() {
        let val = item
            .as_f64()
            .ok_or_else(|| JsValue::from_str(&format!("Expected number at index {}", idx)))?;
        if !val.is_finite() {
            return Err(JsValue::from_str(&format!(
                "Value must be finite at index {}, got {}",
                idx, val
            )));
        }
        vec.push(val as f32);
    }
    Ok(vec)
}

/// Helper to convert JS array of arrays to Vec<Vec<f32>>
#[cfg(feature = "wasm")]
fn js_to_vec_vec_f32(js: &JsValue) -> Result<Vec<Vec<f32>>, JsValue> {
    use wasm_bindgen::JsCast;

    let array = js
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array"))?;

    let mut vecs = Vec::with_capacity(array.length() as usize);
    for (idx, item) in array.iter().enumerate() {
        let vec = js_to_vec_f32(&item).map_err(|e| {
            JsValue::from_str(&format!("Invalid vector at index {}: {:?}", idx, e))
        })?;
        vecs.push(vec);
    }
    Ok(vecs)
}

/// Helper to convert JS array of [id, score] pairs to Vec<(String, f32)>
#[cfg(feature = "wasm")]
fn js_to_candidates(js: &JsValue) -> Result<Vec<(String, f32)>, JsValue> {
    use wasm_bindgen::JsCast;

    let array = js
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array"))?;

    let mut candidates = Vec::with_capacity(array.length() as usize);
    for (idx, item) in array.iter().enumerate() {
        let pair = item
            .dyn_ref::<js_sys::Array>()
            .ok_or_else(|| {
                JsValue::from_str(&format!("Expected [id, score] pair at index {}", idx))
            })?;
        if pair.length() != 2 {
            return Err(JsValue::from_str(&format!(
                "Expected [id, score] pair at index {}, got array of length {}",
                idx,
                pair.length()
            )));
        }
        let id = pair
            .get(0)
            .as_string()
            .ok_or_else(|| JsValue::from_str(&format!("id must be a string at index {}", idx)))?;
        let score_val = pair
            .get(1)
            .as_f64()
            .ok_or_else(|| {
                JsValue::from_str(&format!("score must be a number at index {}", idx))
            })?;
        if !score_val.is_finite() {
            return Err(JsValue::from_str(&format!(
                "score must be finite at index {}, got {}",
                idx, score_val
            )));
        }
        candidates.push((id, score_val as f32));
    }
    Ok(candidates)
}

/// Helper to convert Vec<(String, f32)> to JS array
#[cfg(feature = "wasm")]
fn candidates_to_js(candidates: &[(String, f32)]) -> JsValue {
    let array = js_sys::Array::new();
    for (id, score) in candidates {
        let pair = js_sys::Array::new();
        pair.push(&JsValue::from_str(id));
        pair.push(&JsValue::from_f64(*score as f64));
        array.push(&pair);
    }
    array.into()
}

/// Helper to convert Vec<Vec<f32>> to JS array
#[cfg(feature = "wasm")]
fn vec_vec_f32_to_js(vecs: &[Vec<f32>]) -> JsValue {
    let array = js_sys::Array::new();
    for vec in vecs {
        let inner = js_sys::Array::new();
        for val in vec {
            inner.push(&JsValue::from_f64(*val as f64));
        }
        array.push(&inner);
    }
    array.into()
}

/// Cosine similarity between two vectors.
///
/// # Arguments
/// * `a` - First vector as array of numbers
/// * `b` - Second vector as array of numbers (must have same length as `a`)
///
/// # Returns
/// Cosine similarity score (0.0 to 1.0, or NaN if vectors are zero)
///
/// # Errors
/// Returns error if vectors have different lengths or contain non-finite values.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn cosine(a: &JsValue, b: &JsValue) -> Result<f64, JsValue> {
    let vec_a = js_to_vec_f32(a)?;
    let vec_b = js_to_vec_f32(b)?;

    if vec_a.len() != vec_b.len() {
        return Err(JsValue::from_str(&format!(
            "Vectors must have same length: {} vs {}",
            vec_a.len(),
            vec_b.len()
        )));
    }

    Ok(simd::cosine(&vec_a, &vec_b) as f64)
}

/// MaxSim (ColBERT) score between query and document token embeddings.
///
/// # Arguments
/// * `query_tokens` - Query token embeddings as array of arrays
/// * `doc_tokens` - Document token embeddings as array of arrays
///
/// # Returns
/// MaxSim score (sum of max similarities for each query token)
///
/// # Errors
/// Returns error if input format is invalid.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn maxsim(query_tokens: &JsValue, doc_tokens: &JsValue) -> Result<f64, JsValue> {
    let query = js_to_vec_vec_f32(query_tokens)?;
    let doc = js_to_vec_vec_f32(doc_tokens)?;

    let query_refs: Vec<&[f32]> = query.iter().map(|v| v.as_slice()).collect();
    let doc_refs: Vec<&[f32]> = doc.iter().map(|v| v.as_slice()).collect();

    Ok(simd::maxsim(&query_refs, &doc_refs) as f64)
}

/// Pool tokens using hierarchical clustering (ColBERT).
///
/// # Arguments
/// * `tokens` - Token embeddings as array of arrays
/// * `pool_factor` - Pooling factor (e.g., 2 = reduce to half, must be >= 1)
///
/// # Returns
/// Pooled token embeddings as array of arrays
///
/// # Errors
/// Returns error if `pool_factor` is 0 or if input format is invalid.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn pool_tokens(tokens: &JsValue, pool_factor: usize) -> Result<JsValue, JsValue> {
    let tokens_vec = js_to_vec_vec_f32(tokens)?;

    if pool_factor == 0 {
        return Err(JsValue::from_str("pool_factor must be >= 1"));
    }

    let pooled = colbert::pool_tokens(&tokens_vec, pool_factor)
        .map_err(|e| JsValue::from_str(&format!("Pooling error: {:?}", e)))?;

    Ok(vec_vec_f32_to_js(&pooled))
}

/// Rank documents by MaxSim score.
///
/// # Arguments
/// * `query` - Query token embeddings as array of arrays
/// * `docs` - Array of `[id, tokens]` pairs where `id` is string and `tokens` is array of arrays
/// * `top_k` - Optional limit on number of results (default: None = all)
///
/// # Returns
/// Ranked candidates as array of `[id, score]` pairs
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rank(
    query: &JsValue,
    docs: &JsValue,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    use wasm_bindgen::JsCast;

    let query_vec = js_to_vec_vec_f32(query)?;
    let docs_array = docs
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array of [id, tokens] pairs"))?;

    let mut docs_vec: Vec<(String, Vec<Vec<f32>>)> = Vec::new();
    for (idx, item) in docs_array.iter().enumerate() {
        let pair = item
            .dyn_ref::<js_sys::Array>()
            .ok_or_else(|| {
                JsValue::from_str(&format!("Expected [id, tokens] pair at index {}", idx))
            })?;
        if pair.length() != 2 {
            return Err(JsValue::from_str(&format!(
                "Expected [id, tokens] pair at index {}, got array of length {}",
                idx,
                pair.length()
            )));
        }
        let id = pair
            .get(0)
            .as_string()
            .ok_or_else(|| JsValue::from_str(&format!("id must be a string at index {}", idx)))?;
        let tokens = js_to_vec_vec_f32(&pair.get(1))
            .map_err(|e| JsValue::from_str(&format!("Invalid tokens at index {}: {:?}", idx, e)))?;
        docs_vec.push((id, tokens));
    }

    let ranked = colbert::rank_with_top_k(&query_vec, &docs_vec, top_k);
    Ok(candidates_to_js(&ranked))
}

/// MMR (Maximal Marginal Relevance) diversity selection.
///
/// # Arguments
/// * `candidates` - Array of `[id, score]` pairs
/// * `similarity` - Flattened similarity matrix (row-major, n×n for n candidates)
/// * `k` - Number of items to select
/// * `lambda` - Diversity vs relevance trade-off (0.0 = all relevance, 1.0 = all diversity)
///
/// # Returns
/// Selected diverse candidates as array of `[id, score]` pairs
///
/// # Errors
/// Returns error if similarity matrix size doesn't match candidates or if input format is invalid.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn mmr(
    candidates: &JsValue,
    similarity: &JsValue,
    k: usize,
    lambda: f32,
) -> Result<JsValue, JsValue> {
    let candidates_vec = js_to_candidates(candidates)?;
    let similarity_vec = js_to_vec_f32(similarity)?;

    let n = candidates_vec.len();
    let expected_size = n * n;
    if similarity_vec.len() != expected_size {
        return Err(JsValue::from_str(&format!(
            "Similarity matrix must be n×n ({}×{}={}), got {} elements",
            n, n, expected_size, similarity_vec.len()
        )));
    }

    if !lambda.is_finite() || lambda < 0.0 || lambda > 1.0 {
        return Err(JsValue::from_str("lambda must be between 0.0 and 1.0"));
    }

    let config = MmrConfig::default().with_k(k).with_lambda(lambda);
    let selected = diversity::mmr(&candidates_vec, &similarity_vec, config);
    Ok(candidates_to_js(&selected))
}

/// DPP (Determinantal Point Process) diversity selection.
///
/// # Arguments
/// * `candidates` - Array of `[id, score]` pairs
/// * `embeddings` - Array of embeddings (one per candidate) as arrays of numbers
/// * `k` - Number of items to select
///
/// # Returns
/// Selected diverse candidates as array of `[id, score]` pairs
///
/// # Errors
/// Returns error if embeddings count doesn't match candidates or if input format is invalid.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn dpp(
    candidates: &JsValue,
    embeddings: &JsValue,
    k: usize,
) -> Result<JsValue, JsValue> {
    let candidates_vec = js_to_candidates(candidates)?;
    let embeddings_vec = js_to_vec_vec_f32(embeddings)?;

    if candidates_vec.len() != embeddings_vec.len() {
        return Err(JsValue::from_str(&format!(
            "Candidates and embeddings must have same length: {} vs {}",
            candidates_vec.len(),
            embeddings_vec.len()
        )));
    }

    let config = DppConfig::default().with_k(k);
    let selected = diversity::dpp(&candidates_vec, &embeddings_vec, config);
    Ok(candidates_to_js(&selected))
}

/// Matryoshka refinement using tail dimensions.
///
/// # Arguments
/// * `candidates` - Initial candidates as array of `[id, score]` pairs
/// * `query` - Query embedding as array of numbers
/// * `docs` - Array of `[id, embedding]` pairs where `id` is string and `embedding` is array of numbers
/// * `head_dims` - Number of head dimensions to use (must be < query.len())
///
/// # Returns
/// Refined candidates as array of `[id, score]` pairs
///
/// # Errors
/// Returns error if `head_dims >= query.len()` or if input format is invalid.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn refine(
    candidates: &JsValue,
    query: &JsValue,
    docs: &JsValue,
    head_dims: usize,
) -> Result<JsValue, JsValue> {
    use wasm_bindgen::JsCast;

    let candidates_vec = js_to_candidates(candidates)?;
    let query_vec = js_to_vec_f32(query)?;

    if head_dims >= query_vec.len() {
        return Err(JsValue::from_str(&format!(
            "head_dims ({}) must be < query.len() ({})",
            head_dims,
            query_vec.len()
        )));
    }

    let docs_array = docs
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array of [id, embedding] pairs"))?;

    let mut docs_vec: Vec<(String, Vec<f32>)> = Vec::new();
    for (idx, item) in docs_array.iter().enumerate() {
        let pair = item
            .dyn_ref::<js_sys::Array>()
            .ok_or_else(|| {
                JsValue::from_str(&format!("Expected [id, embedding] pair at index {}", idx))
            })?;
        if pair.length() != 2 {
            return Err(JsValue::from_str(&format!(
                "Expected [id, embedding] pair at index {}, got array of length {}",
                idx,
                pair.length()
            )));
        }
        let id = pair
            .get(0)
            .as_string()
            .ok_or_else(|| JsValue::from_str(&format!("id must be a string at index {}", idx)))?;
        let embedding = js_to_vec_f32(&pair.get(1))
            .map_err(|e| JsValue::from_str(&format!("Invalid embedding at index {}: {:?}", idx, e)))?;
        docs_vec.push((id, embedding));
    }

    let refined = matryoshka::try_refine(
        &candidates_vec,
        &query_vec,
        &docs_vec,
        head_dims,
        RefineConfig::default(),
    )
    .map_err(|e| JsValue::from_str(&format!("Refinement error: {:?}", e)))?;

    Ok(candidates_to_js(&refined))
}

/// MaxSim alignments for interpretability (token-level matching).
///
/// # Arguments
/// * `query_tokens` - Query token embeddings as array of arrays
/// * `doc_tokens` - Document token embeddings as array of arrays
/// * `top_k` - Optional limit on number of alignments per query token (default: 1)
///
/// # Returns
/// Array of `[query_idx, doc_idx, score]` tuples for each alignment
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn maxsim_alignments(
    query_tokens: &JsValue,
    doc_tokens: &JsValue,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    let query = js_to_vec_vec_f32(query_tokens)?;
    let doc = js_to_vec_vec_f32(doc_tokens)?;

    let query_refs: Vec<&[f32]> = query.iter().map(|v| v.as_slice()).collect();
    let doc_refs: Vec<&[f32]> = doc.iter().map(|v| v.as_slice()).collect();

    let alignments = simd::maxsim_alignments(&query_refs, &doc_refs);
    let _top_k = top_k.unwrap_or(1); // Currently unused, reserved for future top-k filtering

    let result = js_sys::Array::new();
    for alignment in alignments.iter() {
        // alignment is (query_idx, doc_idx, score)
        let tuple = js_sys::Array::new();
        tuple.push(&JsValue::from_f64(alignment.0 as f64));
        tuple.push(&JsValue::from_f64(alignment.1 as f64));
        tuple.push(&JsValue::from_f64(alignment.2 as f64));
        result.push(&tuple);
    }

    Ok(result.into())
}

/// Batch MaxSim: score a query against multiple documents.
///
/// # Arguments
/// * `query` - Query token embeddings as array of arrays
/// * `docs` - Array of document token embeddings (each doc is array of arrays)
///
/// # Returns
/// Array of scores, one per document
///
/// # Errors
/// Returns error if input format is invalid.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn maxsim_batch(
    query: &JsValue,
    docs: &JsValue,
) -> Result<JsValue, JsValue> {
    let query_vec = js_to_vec_vec_f32(query)?;
    let docs_array = docs
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array of document token arrays"))?;

    let mut docs_vec: Vec<Vec<Vec<f32>>> = Vec::new();
    for (idx, item) in docs_array.iter().enumerate() {
        let doc = js_to_vec_vec_f32(&item)
            .map_err(|e| JsValue::from_str(&format!("Invalid document at index {}: {:?}", idx, e)))?;
        docs_vec.push(doc);
    }

    let scores = simd::maxsim_batch(&query_vec, &docs_vec);
    let result = js_sys::Array::new();
    for score in scores.iter() {
        result.push(&JsValue::from_f64(*score as f64));
    }

    Ok(result.into())
}

/// Batch token alignments for multiple documents.
///
/// # Arguments
/// * `query` - Query token embeddings as array of arrays
/// * `docs` - Array of document token embeddings (each doc is array of arrays)
///
/// # Returns
/// Array of alignment arrays, one per document. Each alignment is array of `[query_idx, doc_idx, score]` tuples.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn maxsim_alignments_batch(
    query: &JsValue,
    docs: &JsValue,
) -> Result<JsValue, JsValue> {
    let query_vec = js_to_vec_vec_f32(query)?;
    let docs_array = docs
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array of document token arrays"))?;

    let mut docs_vec: Vec<Vec<Vec<f32>>> = Vec::new();
    for (idx, item) in docs_array.iter().enumerate() {
        let doc = js_to_vec_vec_f32(&item)
            .map_err(|e| JsValue::from_str(&format!("Invalid document at index {}: {:?}", idx, e)))?;
        docs_vec.push(doc);
    }

    let alignments = simd::maxsim_alignments_batch(&query_vec, &docs_vec);
    let result = js_sys::Array::new();
    for doc_alignments in alignments.iter() {
        let doc_array = js_sys::Array::new();
        for alignment in doc_alignments.iter() {
            let tuple = js_sys::Array::new();
            tuple.push(&JsValue::from_f64(alignment.0 as f64));
            tuple.push(&JsValue::from_f64(alignment.1 as f64));
            tuple.push(&JsValue::from_f64(alignment.2 as f64));
            doc_array.push(&tuple);
        }
        result.push(&doc_array);
    }

    Ok(result.into())
}

