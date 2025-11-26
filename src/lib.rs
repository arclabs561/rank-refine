//! # rank-refine
//!
//! Model-based reranking for retrieval pipelines.
//!
//! This crate provides refinement/reranking algorithms that take a list of
//! candidates and re-score them using more expensive (but more accurate) methods.
//!
//! ## Planned Features
//!
//! - **Cross-encoder reranking** — Score pairs with transformer models
//! - **Matryoshka refinement** — Refine using tail dimensions of MRL embeddings
//! - **ColBERT/PLAID MaxSim** — Late interaction scoring
//!
//! ## Relationship to rank-fusion
//!
//! ```text
//! Retrieve (BM25, Dense) → Fuse (rank-fusion) → Rerank (rank-refine) → Top-K
//! ```
//!
//! - `rank-fusion`: Combines multiple result lists (zero deps, pure algorithms)
//! - `rank-refine`: Re-scores a single list with expensive models (this crate)
//!
//! ## Example (future API)
//!
//! ```ignore
//! use rank_refine::{MatryoshkaRefiner, Refiner};
//!
//! let refiner = MatryoshkaRefiner::new(64); // coarse = first 64 dims
//! let refined = refiner.refine(&candidates, &query_full_vector);
//! ```

// TODO: Implement refiners
// - [ ] MatryoshkaRefiner (SIMD vector math, no model)
// - [ ] CrossEncoderRefiner (candle backend)
// - [ ] MaxSimRefiner (ColBERT-style)

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

// Placeholder for future implementations

