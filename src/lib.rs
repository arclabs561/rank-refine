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

// ─────────────────────────────────────────────────────────────────────────────
// Newtypes for Type Safety
// ─────────────────────────────────────────────────────────────────────────────

/// A single dense embedding vector.
#[derive(Debug, Clone)]
pub struct Embedding(pub Vec<f32>);

impl AsRef<[f32]> for Embedding {
    fn as_ref(&self) -> &[f32] {
        &self.0
    }
}

impl From<Vec<f32>> for Embedding {
    fn from(v: Vec<f32>) -> Self {
        Self(v)
    }
}

/// Token-level embeddings for ColBERT-style models.
#[derive(Debug, Clone)]
pub struct TokenEmbeddings(pub Vec<Vec<f32>>);

impl TokenEmbeddings {
    /// Number of tokens.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get token embedding dimension (0 if empty).
    #[must_use]
    pub fn dim(&self) -> usize {
        self.0.first().map_or(0, Vec::len)
    }
}

impl From<Vec<Vec<f32>>> for TokenEmbeddings {
    fn from(v: Vec<Vec<f32>>) -> Self {
        Self(v)
    }
}

impl AsRef<[Vec<f32>]> for TokenEmbeddings {
    fn as_ref(&self) -> &[Vec<f32>] {
        &self.0
    }
}
