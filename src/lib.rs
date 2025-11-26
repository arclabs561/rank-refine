//! Reranking for retrieval pipelines.
//!
//! This crate provides **scoring algorithms only** — no model weights, no inference.
//! You bring your own embeddings (from any source: sentence-transformers, fastembed,
//! candle, ONNX, etc.) and this crate handles the reranking math.
//!
//! ## Modules
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`matryoshka`] | Refine using MRL tail dimensions |
//! | [`colbert`] | `MaxSim` late interaction + token pooling |
//! | [`crossencoder`] | Cross-encoder trait (implement for your model) |
//! | [`simd`] | Vector ops (AVX2/NEON accelerated) |
//! | [`scoring`] | Unified [`Scorer`](scoring::Scorer), [`TokenScorer`](scoring::TokenScorer), [`Pooler`](scoring::Pooler) traits |
//!
//! ## Quick Start
//!
//! ```rust
//! use rank_refine::{simd, colbert, matryoshka};
//!
//! // Dense scoring (your embeddings, our math)
//! let score = simd::cosine(&[1.0, 0.0], &[0.707, 0.707]);
//!
//! // ColBERT MaxSim (token-level embeddings)
//! let query_tokens: Vec<&[f32]> = vec![&[1.0, 0.0], &[0.0, 1.0]];
//! let doc_tokens: Vec<&[f32]> = vec![&[0.9, 0.1], &[0.1, 0.9]];
//! let maxsim_score = simd::maxsim(&query_tokens, &doc_tokens);
//!
//! // Token pooling (compress 32 tokens to 8)
//! let tokens: Vec<Vec<f32>> = vec![vec![1.0; 128]; 32];
//! let pooled = colbert::pool_tokens_adaptive(&tokens, 4); // factor 4 -> 8 tokens
//! ```
//!
//! ## Design: Bring Your Own Model (BYOM)
//!
//! This crate is intentionally **model-agnostic**:
//!
//! - **No downloads**: Tests use synthetic vectors, not real models
//! - **No dependencies**: No ONNX, PyTorch, or ML frameworks
//! - **Pure Rust**: SIMD-accelerated math you can audit
//!
//! For cross-encoders, implement the [`crossencoder::CrossEncoderModel`] trait:
//!
//! ```rust
//! use rank_refine::crossencoder::CrossEncoderModel;
//!
//! struct MyModel; // Your inference implementation
//!
//! impl CrossEncoderModel for MyModel {
//!     fn score_batch(&self, query: &str, documents: &[&str]) -> Vec<f32> {
//!         // Call your ONNX/candle/tch model here
//!         documents.iter().map(|_| 0.5).collect() // placeholder
//!     }
//! }
//! ```
//!
//! ## Complete Pipeline Example
//!
//! Here's a realistic two-stage retrieval pipeline:
//!
//! ```rust
//! use rank_refine::{simd, colbert, scoring::{Scorer, DenseScorer, AdaptivePooler, Pooler}};
//!
//! // Stage 1: Fast dense retrieval (your embeddings, our math)
//! let query_dense: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4]; // from your embedding model
//! let docs_dense = vec![
//!     ("doc1", vec![0.15, 0.25, 0.35, 0.45]),
//!     ("doc2", vec![0.9, 0.1, 0.0, 0.0]),
//! ];
//!
//! let scorer = DenseScorer::Cosine;
//! let doc_refs: Vec<(&str, &[f32])> = docs_dense.iter()
//!     .map(|(id, emb)| (*id, emb.as_slice()))
//!     .collect();
//! let first_stage: Vec<(&str, f32)> = scorer.rank(&query_dense, &doc_refs);
//!
//! // Stage 2: Refine top-k with ColBERT (late interaction)
//! let query_tokens: Vec<Vec<f32>> = vec![
//!     vec![0.1, 0.9, 0.0, 0.0], // token 1
//!     vec![0.0, 0.1, 0.8, 0.1], // token 2
//! ];
//! let doc_tokens = vec![
//!     ("doc1", vec![vec![0.2, 0.8, 0.1, 0.0], vec![0.0, 0.2, 0.7, 0.1]]),
//!     ("doc2", vec![vec![0.9, 0.1, 0.0, 0.0]]),
//! ];
//!
//! // Optional: compress token embeddings for storage efficiency
//! let pooler = AdaptivePooler;
//! let doc1_pooled = pooler.pool_by_factor(&doc_tokens[0].1, 2);
//!
//! // Compute MaxSim score
//! let q_refs: Vec<&[f32]> = query_tokens.iter().map(Vec::as_slice).collect();
//! let d_refs: Vec<&[f32]> = doc_tokens[0].1.iter().map(Vec::as_slice).collect();
//! let maxsim = simd::maxsim(&q_refs, &d_refs);
//! assert!(maxsim > 0.0);
//! ```
//!
//! ## Error Handling
//!
//! Functions return [`Result<T, RefineError>`] for invalid inputs.
//! Use the `try_*` variants for fallible operations.

pub mod colbert;
pub mod crossencoder;
pub mod matryoshka;
pub mod scoring;
pub mod simd;

// ─────────────────────────────────────────────────────────────────────────────
// Error Types
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during refinement.
#[derive(Debug, Clone, PartialEq)]
pub enum RefineError {
    /// Head dimensions >= query length (no tail to refine with).
    InvalidHeadDims { head_dims: usize, query_len: usize },
    /// Empty query (cannot compute similarity).
    EmptyQuery,
    /// Dimension mismatch between query and document.
    DimensionMismatch { expected: usize, got: usize },
}

impl std::fmt::Display for RefineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidHeadDims {
                head_dims,
                query_len,
            } => {
                write!(
                    f,
                    "head_dims ({head_dims}) must be < query.len() ({query_len})"
                )
            }
            Self::EmptyQuery => write!(f, "query is empty"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for RefineError {}

/// Result type for refinement operations.
pub type Result<T> = std::result::Result<T, RefineError>;

// ─────────────────────────────────────────────────────────────────────────────
// Conversion Utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Convert owned token embeddings to borrowed slices.
///
/// This is the idiomatic pattern for working with the SIMD functions:
///
/// ```rust
/// use rank_refine::{simd, as_slices};
///
/// let tokens: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
/// let refs = as_slices(&tokens);
/// // Now you can pass `&refs` to simd::maxsim, scoring traits, etc.
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

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for refinement operations.
///
/// # Example
///
/// ```rust
/// use rank_refine::RefineConfig;
///
/// let config = RefineConfig::default()
///     .with_alpha(0.7)
///     .with_top_k(10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefineConfig {
    /// Weight for original score (0.0 = all new, 1.0 = all original).
    pub alpha: f32,
    /// Maximum results to return (None = all).
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
    /// Set the blending weight (0.0 = all refinement, 1.0 = all original).
    #[must_use]
    pub const fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Limit output to `top_k` results.
    #[must_use]
    pub const fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Use only refinement scores (ignore original).
    #[must_use]
    pub const fn refinement_only() -> Self {
        Self {
            alpha: 0.0,
            top_k: None,
        }
    }

    /// Use only original scores (no refinement).
    #[must_use]
    pub const fn original_only() -> Self {
        Self {
            alpha: 1.0,
            top_k: None,
        }
    }
}
