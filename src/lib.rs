//! # rank-refine
//!
//! Reranking algorithms for retrieval pipelines.
//!
//! Takes candidates from initial retrieval and re-scores them using
//! more expensive methods.
//!
//! ## Modules
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`matryoshka`] | Refine using MRL tail dimensions |
//! | [`colbert`] | MaxSim late interaction scoring |
//! | [`crossencoder`] | Cross-encoder trait (BYOM) |
//! | [`simd`] | Vector ops with AVX2/NEON dispatch |
//!
//! ## Pipeline
//!
//! ```text
//! Retrieve → Fuse (rank-fusion) → Refine (this crate) → Top-K
//! ```
//!
//! ## Example
//!
//! ```rust
//! use rank_refine::matryoshka;
//!
//! let candidates = vec![("doc1", 0.9), ("doc2", 0.8)];
//! let query = vec![0.1; 128];
//! let docs = vec![
//!     ("doc1", vec![0.2; 128]),
//!     ("doc2", vec![0.15; 128]),
//! ];
//!
//! let refined = matryoshka::refine(&candidates, &query, &docs, 64);
//! ```

pub mod colbert;
pub mod crossencoder;
pub mod matryoshka;
pub mod simd;
