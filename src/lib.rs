//! # rank-refine
//!
//! Model-based reranking for retrieval pipelines.
//!
//! This crate provides refinement/reranking algorithms that take a list of
//! candidates and re-score them using more expensive (but more accurate) methods.
//!
//! ## Available Refiners
//!
//! - [`matryoshka::refine`] — Refine using tail dimensions of MRL embeddings
//!
//! ## Relationship to rank-fusion
//!
//! ```text
//! Retrieve (BM25, Dense) → Fuse (rank-fusion) → Refine (this crate) → Top-K
//! ```
//!
//! - `rank-fusion`: Combines multiple result lists (zero deps, pure algorithms)
//! - `rank-refine`: Re-scores a single list with expensive methods (this crate)
//!
//! ## Example
//!
//! ```rust
//! use rank_refine::matryoshka;
//!
//! // Candidates from initial retrieval (using first 64 dims)
//! let candidates = vec![("doc1", 0.9), ("doc2", 0.85), ("doc3", 0.7)];
//!
//! // Full embeddings for refinement
//! let query: Vec<f32> = vec![0.1; 128];  // 128-dim query
//! let docs: Vec<(&str, Vec<f32>)> = vec![
//!     ("doc1", vec![0.2; 128]),
//!     ("doc2", vec![0.15; 128]),
//!     ("doc3", vec![0.1; 128]),
//! ];
//!
//! // Refine using dimensions 64..128 (the "tail")
//! let refined = matryoshka::refine(&candidates, &query, &docs, 64);
//! ```

pub mod matryoshka;

/// Trait for refinement/reranking algorithms.
///
/// A `Refiner` takes a list of candidates (from initial retrieval + fusion)
/// and produces a re-scored list using a more expensive scoring method.
pub trait Refiner<I, Q> {
    /// Refine the candidate list using the query context.
    ///
    /// Returns candidates re-scored and sorted by descending score.
    fn refine(&self, candidates: &[(I, f32)], query: &Q) -> Vec<(I, f32)>;
}
