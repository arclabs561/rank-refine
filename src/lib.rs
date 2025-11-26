//! Reranking for retrieval pipelines.
//!
//! - [`matryoshka`] — Refine using MRL tail dimensions
//! - [`colbert`] — MaxSim late interaction
//! - [`crossencoder`] — Cross-encoder trait (BYOM)
//! - [`simd`] — Vector ops (AVX2/NEON)

pub mod colbert;
pub mod crossencoder;
pub mod matryoshka;
pub mod simd;
