//! # rank-refine
//!
//! Fast reranking for retrieval pipelines.
//!
//! ## Modules
//!
//! | Module | Purpose | Notes |
//! |--------|---------|-------|
//! | [`matryoshka`] | Refine with MRL tail dimensions | Zero deps |
//! | [`colbert`] | MaxSim late interaction | Zero deps |
//! | [`crossencoder`] | Transformer scoring | Trait-based, BYOM |
//! | [`simd`] | Vector ops (AVX2/NEON) | Auto-dispatch |
//!
//! ## Pipeline
//!
//! ```text
//! Retrieve → Fuse (rank-fusion) → Refine (this crate) → Top-K
//! ```
//!
//! ## Performance
//!
//! SIMD-accelerated on x86_64 (AVX2+FMA) and aarch64 (NEON):
//! - **3×** faster dot/cosine operations
//! - **3.8×** faster MaxSim scoring
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
pub mod crossencoder;
pub mod matryoshka;
pub mod simd;
