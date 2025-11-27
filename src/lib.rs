//! Rerank search candidates with embeddings. SIMD-accelerated.
//!
//! You bring embeddings, this crate scores them. No model weights, no inference.
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
//! | Module | Purpose |
//! |--------|---------|
//! | [`simd`] | SIMD vector ops (dot, cosine, maxsim) |
//! | [`colbert`] | Late interaction (MaxSim), token pooling |
//! | [`diversity`] | MMR + DPP diversity selection |
//! | [`crossencoder`] | Cross-encoder trait |
//! | [`matryoshka`] | MRL tail refinement |
//!
//! Advanced: [`scoring`] for trait-based polymorphism, [`embedding`] for type-safe wrappers.

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
    // Core SIMD functions
    pub use crate::simd::{cosine, cosine_truncating, dot, dot_truncating, maxsim, norm};

    // Score utilities
    pub use crate::simd::{normalize_maxsim, softmax_scores, top_k_indices};

    // ColBERT
    pub use crate::colbert::{pool_tokens, rank as colbert_rank};

    // Diversity
    pub use crate::diversity::{dpp, mmr, DppConfig, MmrConfig};

    // Cross-encoder trait
    pub use crate::crossencoder::CrossEncoderModel;
}

// ─────────────────────────────────────────────────────────────────────────────
// Error Types
// ─────────────────────────────────────────────────────────────────────────────

/// Errors from refinement operations.
#[derive(Debug, Clone, PartialEq)]
pub enum RefineError {
    /// `head_dims` must be less than `query.len()` for tail refinement.
    InvalidHeadDims {
        /// The head dimension that was provided.
        head_dims: usize,
        /// The query length (must be > head_dims).
        query_len: usize,
    },
    /// Vector dimensions must match.
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension received.
        got: usize,
    },
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
    /// Set blending weight. Clamped to \[0, 1\].
    ///
    /// - `0.0` = all refinement score
    /// - `0.5` = equal blend (default)
    /// - `1.0` = all original score
    #[must_use]
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refine_config_clamps_alpha() {
        assert_eq!(RefineConfig::default().with_alpha(-0.5).alpha, 0.0);
        assert_eq!(RefineConfig::default().with_alpha(1.5).alpha, 1.0);
        assert_eq!(RefineConfig::default().with_alpha(0.7).alpha, 0.7);
    }
}
