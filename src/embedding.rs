//! Type-safe embedding wrappers that encode geometric invariants.
//!
//! This module provides newtypes that use the type system to prevent
//! misuse and encode mathematical properties at compile time.
//!
//! ## Training Paradigm Taxonomy
//!
//! | Paradigm | Loss Function | Resulting Geometry | Scoring |
//! |----------|---------------|--------------------|---------| 
//! | Contrastive | InfoNCE, triplet | Unit hypersphere | Cosine = Dot |
//! | Metric Learning | Pull-push | Learned metric | Euclidean or learned |
//! | Instruction-tuned | Task-conditioned contrastive | Asymmetric query/doc | Cosine |
//!
//! ## Type Safety
//!
//! The type system encodes:
//! - **Normalization**: `Normalized<T>` guarantees unit L2 norm
//! - **Role**: `Query<T>` vs `Document<T>` for asymmetric models
//! - **Masking**: `Masked<T>` carries validity information for padding
//!
//! ## Example
//!
//! ```rust
//! use rank_refine::embedding::{Normalized, normalize};
//!
//! // Compile-time guarantee: these are unit vectors
//! let q = normalize(&[3.0, 4.0]).unwrap();
//! let d = normalize(&[1.0, 0.0]).unwrap();
//!
//! // Dot product on normalized vectors IS cosine similarity
//! let cosine = q.dot(&d);
//! ```

use std::marker::PhantomData;

// ─────────────────────────────────────────────────────────────────────────────
// Normalized Embedding (unit L2 norm)
// ─────────────────────────────────────────────────────────────────────────────

/// A unit-normalized embedding vector.
///
/// # Invariant
///
/// `||v||₂ = 1` (within floating-point tolerance)
///
/// # Why This Matters
///
/// Contrastive learning (SimCLR, InfoNCE) produces embeddings where
/// `cos(q, d) = q · d` because both are unit normalized. This type
/// makes that assumption explicit and compiler-checked.
#[derive(Debug, Clone)]
pub struct Normalized {
    data: Vec<f32>,
}

impl Normalized {
    /// Access the underlying data (guaranteed unit norm).
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Dimension of the embedding.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// Dot product with another normalized vector.
    ///
    /// Because both are unit normalized, this IS cosine similarity.
    #[inline]
    #[must_use]
    pub fn dot(&self, other: &Normalized) -> f32 {
        crate::simd::dot(&self.data, &other.data)
    }

    /// Cosine similarity (redundant for normalized, but explicit).
    #[inline]
    #[must_use]
    pub fn cosine(&self, other: &Normalized) -> f32 {
        // For normalized vectors, dot = cosine
        self.dot(other)
    }
}

/// Normalize a vector to unit L2 norm.
///
/// Returns `None` if the vector is zero (undefined normalization).
#[must_use]
pub fn normalize(v: &[f32]) -> Option<Normalized> {
    let norm = crate::simd::norm(v);
    if norm < 1e-9 {
        return None;
    }
    let data: Vec<f32> = v.iter().map(|&x| x / norm).collect();
    Some(Normalized { data })
}

/// Normalize a vector, defaulting to zero vector if norm is zero.
///
/// Use when you need a value even for degenerate inputs.
#[must_use]
pub fn normalize_or_zero(v: &[f32]) -> Normalized {
    normalize(v).unwrap_or_else(|| Normalized {
        data: vec![0.0; v.len()],
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Query/Document Role (for asymmetric models)
// ─────────────────────────────────────────────────────────────────────────────

/// Marker for query embeddings.
pub struct QueryRole;

/// Marker for document embeddings.
pub struct DocumentRole;

/// An embedding with explicit role (query or document).
///
/// # Why This Matters
///
/// Instruction-tuned models (INSTRUCTOR, E5) use different prefixes
/// for queries vs documents. This type prevents accidentally scoring
/// a query against a query, or mixing up the roles.
///
/// ```text
/// INSTRUCTOR:
///   query:    "Represent the question for retrieval: {text}"
///   document: "Represent the document for retrieval: {text}"
/// ```
#[derive(Debug, Clone)]
pub struct Embed<Role> {
    data: Vec<f32>,
    _role: PhantomData<Role>,
}

/// A query embedding (from instruction-tuned model with query prefix).
pub type QueryEmbed = Embed<QueryRole>;

/// A document embedding (from instruction-tuned model with doc prefix).
pub type DocEmbed = Embed<DocumentRole>;

impl<R> Embed<R> {
    /// Create an embedding with the specified role.
    pub fn new(data: Vec<f32>) -> Self {
        Self {
            data,
            _role: PhantomData,
        }
    }

    /// Access the underlying data.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Dimension of the embedding.
    #[inline]
    #[must_use]
    pub fn dim(&self) -> usize {
        self.data.len()
    }
}

impl QueryEmbed {
    /// Score query against document (the correct direction).
    #[inline]
    #[must_use]
    pub fn score(&self, doc: &DocEmbed) -> f32 {
        crate::simd::cosine(&self.data, &doc.data)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Masked Token Embeddings (for variable-length sequences)
// ─────────────────────────────────────────────────────────────────────────────

/// Token embeddings with explicit padding mask.
///
/// # Why This Matters
///
/// Batched processing (PyLate, ColPali style) requires padding shorter
/// sequences. The mask indicates which tokens are real vs padding.
///
/// # Invariant
///
/// `tokens.len() == mask.len()` and `mask[i]` indicates validity of `tokens[i]`.
#[derive(Debug, Clone)]
pub struct MaskedTokens {
    tokens: Vec<Vec<f32>>,
    mask: Vec<bool>,
}

impl MaskedTokens {
    /// Create masked tokens from tokens and mask.
    ///
    /// # Panics
    ///
    /// Panics if `tokens.len() != mask.len()`.
    pub fn new(tokens: Vec<Vec<f32>>, mask: Vec<bool>) -> Self {
        assert_eq!(
            tokens.len(),
            mask.len(),
            "Token count {} must match mask length {}",
            tokens.len(),
            mask.len()
        );
        Self { tokens, mask }
    }

    /// Create from tokens without any masking (all valid).
    pub fn from_tokens(tokens: Vec<Vec<f32>>) -> Self {
        let mask = vec![true; tokens.len()];
        Self { tokens, mask }
    }

    /// Number of tokens (including padding).
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether there are no tokens.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Number of valid (non-padding) tokens.
    #[inline]
    #[must_use]
    pub fn valid_count(&self) -> usize {
        self.mask.iter().filter(|&&m| m).count()
    }

    /// Iterator over valid tokens only.
    pub fn valid_tokens(&self) -> impl Iterator<Item = &[f32]> {
        self.tokens
            .iter()
            .zip(self.mask.iter())
            .filter_map(|(t, &m)| if m { Some(t.as_slice()) } else { None })
    }

    /// Access all tokens (including padding).
    pub fn all_tokens(&self) -> &[Vec<f32>] {
        &self.tokens
    }

    /// Access mask.
    pub fn mask(&self) -> &[bool] {
        &self.mask
    }
}

/// MaxSim with masking support.
///
/// Ignores padding tokens (where mask is false) in both query and document.
#[must_use]
pub fn maxsim_masked(query: &MaskedTokens, doc: &MaskedTokens) -> f32 {
    if query.valid_count() == 0 || doc.valid_count() == 0 {
        return 0.0;
    }

    // Collect valid doc tokens for repeated iteration
    let valid_doc_tokens: Vec<&[f32]> = doc.valid_tokens().collect();

    query
        .valid_tokens()
        .map(|q| {
            valid_doc_tokens
                .iter()
                .map(|d| crate::simd::dot(q, d))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// Embedding Properties (trait-based encoding)
// ─────────────────────────────────────────────────────────────────────────────

/// Marker trait for embeddings from contrastively trained models.
///
/// Guarantees: unit normalized, symmetric scoring OK.
pub trait ContrastiveEmbed {
    fn as_slice(&self) -> &[f32];
}

impl ContrastiveEmbed for Normalized {
    fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

/// Marker trait for embeddings from instruction-tuned models.
///
/// Guarantees: query/doc roles matter, use asymmetric scoring.
pub trait InstructionEmbed {
    fn as_slice(&self) -> &[f32];
}

impl<R> InstructionEmbed for Embed<R> {
    fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let v = normalize(&[3.0, 4.0]).unwrap();
        let norm: f32 = v.as_slice().iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_zero() {
        assert!(normalize(&[0.0, 0.0, 0.0]).is_none());
    }

    #[test]
    fn test_normalized_dot_is_cosine() {
        let a = normalize(&[1.0, 0.0]).unwrap();
        let b = normalize(&[1.0, 1.0]).unwrap();

        let dot_result = a.dot(&b);
        let cosine_result = a.cosine(&b);

        // For normalized vectors, dot = cosine
        assert!((dot_result - cosine_result).abs() < 1e-6);
    }

    #[test]
    fn test_query_doc_scoring() {
        let query = QueryEmbed::new(vec![1.0, 0.0, 0.0]);
        let doc = DocEmbed::new(vec![0.9, 0.1, 0.0]);

        let score = query.score(&doc);
        assert!(score > 0.9); // should be close to 1
    }

    #[test]
    fn test_masked_tokens() {
        let tokens = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 0.0], // padding
        ];
        let mask = vec![true, true, false];

        let masked = MaskedTokens::new(tokens, mask);
        assert_eq!(masked.len(), 3);
        assert_eq!(masked.valid_count(), 2);
    }

    #[test]
    fn test_maxsim_masked() {
        let query = MaskedTokens::new(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![true, true],
        );
        let doc = MaskedTokens::new(
            vec![vec![1.0, 0.0], vec![0.5, 0.5], vec![0.0, 0.0]], // last is padding
            vec![true, true, false],
        );

        let score = maxsim_masked(&query, &doc);
        // q1 matches d1 perfectly (1.0), q2 matches d2 best (0.5)
        assert!((score - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_maxsim_masked_empty() {
        let query = MaskedTokens::new(vec![vec![1.0, 0.0]], vec![false]); // all masked
        let doc = MaskedTokens::from_tokens(vec![vec![1.0, 0.0]]);

        let score = maxsim_masked(&query, &doc);
        assert!((score - 0.0).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "Token count")]
    fn test_masked_mismatched_lengths() {
        let _ = MaskedTokens::new(vec![vec![1.0]], vec![true, false]);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Normalized vectors have unit norm
        #[test]
        fn normalized_has_unit_norm(v in proptest::collection::vec(-10.0f32..10.0, 2..16)) {
            if let Some(n) = normalize(&v) {
                let norm: f32 = n.as_slice().iter().map(|x| x * x).sum::<f32>().sqrt();
                prop_assert!((norm - 1.0).abs() < 1e-4, "Norm was {}", norm);
            }
        }

        /// Normalized dot is symmetric
        #[test]
        fn normalized_dot_symmetric(
            a in proptest::collection::vec(-10.0f32..10.0, 4..8),
            b in proptest::collection::vec(-10.0f32..10.0, 4..8)
        ) {
            if let (Some(na), Some(nb)) = (normalize(&a), normalize(&b)) {
                let ab = na.dot(&nb);
                let ba = nb.dot(&na);
                prop_assert!((ab - ba).abs() < 1e-5);
            }
        }

        /// Normalized cosine bounded [-1, 1]
        #[test]
        fn normalized_cosine_bounded(
            a in proptest::collection::vec(-10.0f32..10.0, 4..8),
            b in proptest::collection::vec(-10.0f32..10.0, 4..8)
        ) {
            if let (Some(na), Some(nb)) = (normalize(&a), normalize(&b)) {
                let cos = na.cosine(&nb);
                prop_assert!(cos >= -1.01 && cos <= 1.01, "Cosine was {}", cos);
            }
        }

        /// Masked valid_count matches mask
        #[test]
        fn masked_valid_count_matches(
            n_tokens in 1usize..10,
            dim in 2usize..8
        ) {
            let tokens: Vec<Vec<f32>> = (0..n_tokens)
                .map(|i| (0..dim).map(|j| (i + j) as f32 * 0.1).collect())
                .collect();
            let mask: Vec<bool> = (0..n_tokens).map(|i| i % 2 == 0).collect();
            let expected_valid = mask.iter().filter(|&&m| m).count();

            let masked = MaskedTokens::new(tokens, mask);
            prop_assert_eq!(masked.valid_count(), expected_valid);
        }

        /// maxsim_masked with all-valid mask equals regular maxsim
        #[test]
        fn maxsim_masked_all_valid_equals_regular(
            n_q in 1usize..4,
            n_d in 1usize..4,
            dim in 2usize..8
        ) {
            let query_tokens: Vec<Vec<f32>> = (0..n_q)
                .map(|i| (0..dim).map(|j| ((i * dim + j) as f32 * 0.1).sin()).collect())
                .collect();
            let doc_tokens: Vec<Vec<f32>> = (0..n_d)
                .map(|i| (0..dim).map(|j| ((i * dim + j + 1) as f32 * 0.1).cos()).collect())
                .collect();

            let masked_q = MaskedTokens::from_tokens(query_tokens.clone());
            let masked_d = MaskedTokens::from_tokens(doc_tokens.clone());

            let masked_score = maxsim_masked(&masked_q, &masked_d);
            let regular_score = crate::simd::maxsim_vecs(&query_tokens, &doc_tokens);

            prop_assert!((masked_score - regular_score).abs() < 1e-5);
        }
    }
}

