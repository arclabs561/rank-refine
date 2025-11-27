//! Reranking algorithms for retrieval pipelines.
//!
//! You bring embeddings, this crate does the math. No model weights, no inference,
//! no downloads. Get embeddings from fastembed, candle, ort, or serialize from Python.
//!
//! # Quick Start
//!
//! ```rust
//! use rank_refine::simd;
//!
//! // Dense scoring
//! let score = simd::cosine(&[1.0, 0.0], &[0.707, 0.707]);
//!
//! // Late interaction (ColBERT MaxSim)
//! let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
//! let doc = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
//! let score = simd::maxsim_vecs(&query, &doc);
//! ```
//!
//! # Modules
//!
//! - [`simd`] — SIMD-accelerated vector ops (dot, cosine, maxsim)
//! - [`colbert`] — Late interaction ranking and token pooling
//! - [`matryoshka`] — MRL tail dimension refinement
//! - [`crossencoder`] — Cross-encoder trait for transformer models
//! - [`scoring`] — Unified traits for dense/late-interaction scoring
//! - [`embedding`] — Type-safe embedding wrappers (normalized, query/doc roles, masking)
//! - [`diversity`] — MMR and diversity-aware reranking
//!
//! # Cross-Encoder Integration
//!
//! Implement [`crossencoder::CrossEncoderModel`] to use your inference backend:
//!
//! ```rust
//! use rank_refine::crossencoder::CrossEncoderModel;
//!
//! struct MyModel;
//!
//! impl CrossEncoderModel for MyModel {
//!     fn score_batch(&self, query: &str, documents: &[&str]) -> Vec<f32> {
//!         // Your ONNX/candle/tch inference here
//!         vec![0.5; documents.len()]
//!     }
//! }
//! ```
//!
//! # Token Pooling
//!
//! Reduce ColBERT storage by clustering similar tokens:
//!
//! ```rust
//! use rank_refine::colbert;
//!
//! let tokens: Vec<Vec<f32>> = vec![vec![1.0; 128]; 32];
//! let pooled = colbert::pool_tokens(&tokens, 2); // 50% reduction
//! assert!(pooled.len() <= 16);
//! ```
//!
//! For aggressive pooling (4x+), enable the `hierarchical` feature for Ward's method.

pub mod colbert;
pub mod crossencoder;
pub mod diversity;
pub mod embedding;
pub mod matryoshka;
pub mod scoring;
pub mod simd;

/// Common imports for reranking.
///
/// ```rust
/// use rank_refine::prelude::*;
/// ```
pub mod prelude {
    // Type-safe embeddings
    pub use crate::embedding::{maxsim_masked, normalize, MaskedTokens, Normalized};

    // Traits and types
    pub use crate::scoring::{
        DenseScorer, FnPooler, LateInteractionScorer, Pooler, Scorer, TokenScorer,
    };

    // SIMD functions
    pub use crate::simd::{cosine, dot, maxsim, maxsim_cosine, maxsim_vecs, norm};

    // Matryoshka (MRL) refinement
    pub use crate::matryoshka::{refine as mrl_refine, try_refine as mrl_try_refine};
    pub use crate::RefineConfig;

    // ColBERT late interaction
    pub use crate::colbert::{pool_tokens, rank as colbert_rank};

    // Cross-encoder
    pub use crate::crossencoder::CrossEncoderModel;

    // Diversity reranking
    pub use crate::diversity::{mmr, mmr_cosine, try_mmr, MmrConfig};

    // Utilities
    pub use crate::{as_slices, RefineError};
}

// ─────────────────────────────────────────────────────────────────────────────
// Error Types
// ─────────────────────────────────────────────────────────────────────────────

/// Errors from refinement operations.
#[derive(Debug, Clone, PartialEq)]
pub enum RefineError {
    /// `head_dims` must be less than `query.len()` for tail refinement.
    InvalidHeadDims { head_dims: usize, query_len: usize },
    /// Cannot score an empty query.
    EmptyQuery,
    /// Vector dimensions must match.
    DimensionMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for RefineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidHeadDims {
                head_dims,
                query_len,
            } => write!(
                f,
                "invalid head_dims: {head_dims} >= query length {query_len}"
            ),
            Self::EmptyQuery => write!(f, "empty query"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "expected {expected} dimensions, got {got}")
            }
        }
    }
}

impl std::error::Error for RefineError {}

/// Result type for refinement operations.
pub type Result<T> = std::result::Result<T, RefineError>;

/// Convert `&[Vec<f32>]` to `Vec<&[f32]>`.
///
/// Convenience for passing owned token vectors to slice-based APIs:
///
/// ```rust
/// use rank_refine::{simd, as_slices};
///
/// let tokens = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
/// let refs = as_slices(&tokens);
/// let score = simd::maxsim(&refs, &refs);
/// ```
#[inline]
#[must_use]
pub fn as_slices(tokens: &[Vec<f32>]) -> Vec<&[f32]> {
    tokens.iter().map(Vec::as_slice).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Sorting Utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Sort scored results in descending order (highest score first).
///
/// Uses `f32::total_cmp` for deterministic ordering of NaN values.
#[inline]
pub(crate) fn sort_scored_desc<T>(results: &mut [(T, f32)]) {
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
}

/// Configuration for score blending and truncation.
///
/// ```rust
/// use rank_refine::RefineConfig;
///
/// let config = RefineConfig::default()
///     .with_alpha(0.7)  // 70% original, 30% refinement
///     .with_top_k(10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefineConfig {
    /// Blending weight: 0.0 = all refinement, 1.0 = all original. Default: 0.5.
    pub alpha: f32,
    /// Truncate to top k results. Default: None (return all).
    pub top_k: Option<usize>,
}

impl Default for RefineConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            top_k: None,
        }
    }
}

impl RefineConfig {
    /// Set blending weight.
    #[must_use]
    pub const fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Limit output to top k.
    #[must_use]
    pub const fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Only use refinement scores (alpha = 0).
    #[must_use]
    pub const fn refinement_only() -> Self {
        Self {
            alpha: 0.0,
            top_k: None,
        }
    }

    /// Only use original scores (alpha = 1).
    #[must_use]
    pub const fn original_only() -> Self {
        Self {
            alpha: 1.0,
            top_k: None,
        }
    }
}
