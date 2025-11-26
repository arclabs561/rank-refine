//! Reranking for retrieval pipelines.
//!
//! - [`scoring`] — Unified scoring traits (`Scorer`, `TokenScorer`)
//! - [`matryoshka`] — Refine using MRL tail dimensions
//! - [`colbert`] — `MaxSim` late interaction + token pooling
//! - [`crossencoder`] — Cross-encoder trait (BYOM)
//! - [`simd`] — Vector ops (AVX2/NEON)
//!
//! ## Error Handling
//!
//! Functions return [`Result<T, RefineError>`] for invalid inputs.
//! Use the `try_*` variants for fallible operations, or the panicking
//! versions if you've validated inputs.

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
    InvalidHeadDims {
        head_dims: usize,
        query_len: usize,
    },
    /// Empty query (cannot compute similarity).
    EmptyQuery,
    /// Dimension mismatch between query and document.
    DimensionMismatch {
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for RefineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidHeadDims { head_dims, query_len } => {
                write!(f, "head_dims ({head_dims}) must be < query.len() ({query_len})")
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

    /// Limit output to top_k results.
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
