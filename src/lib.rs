//! # rank-refine
//!
//! Reranking for retrieval pipelines.
//!
//! ## Modules
//!
//! - [`matryoshka`] — Refine using tail dimensions of MRL embeddings
//! - [`colbert`] — MaxSim scoring for token-level embeddings
//! - [`simd`] — Vector operations (dot, cosine, maxsim)
//!
//! ## Pipeline
//!
//! ```text
//! Retrieve → Fuse (rank-fusion) → Refine (this crate) → Top-K
//! ```
//!
//! ## Quick Example
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
pub mod matryoshka;
pub mod simd;
