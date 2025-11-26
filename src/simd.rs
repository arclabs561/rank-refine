//! Vector operations (SIMD-ready).
//!
//! Portable implementations that auto-vectorize well.
//! Future: explicit SIMD intrinsics behind feature flag.

/// Dot product of two vectors.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm of a vector.
#[inline]
pub fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// Cosine similarity between two vectors.
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let d = dot(a, b);
    let na = norm(a);
    let nb = norm(b);
    if na > 1e-9 && nb > 1e-9 {
        d / (na * nb)
    } else {
        0.0
    }
}

/// MaxSim: max over query tokens of max similarity to any doc token.
///
/// Used by ColBERT/PLAID for late interaction scoring.
///
/// Formula: `score = Σ_q max_d(q · d)`
///
/// Where q iterates over query token embeddings and d over doc token embeddings.
#[inline]
pub fn maxsim(query_tokens: &[&[f32]], doc_tokens: &[&[f32]]) -> f32 {
    query_tokens
        .iter()
        .map(|q| {
            doc_tokens
                .iter()
                .map(|d| dot(q, d))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

/// MaxSim with cosine similarity instead of dot product.
#[inline]
pub fn maxsim_cosine(query_tokens: &[&[f32]], doc_tokens: &[&[f32]]) -> f32 {
    query_tokens
        .iter()
        .map(|q| {
            doc_tokens
                .iter()
                .map(|d| cosine(q, d))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot() {
        assert!((dot(&[1.0, 2.0], &[3.0, 4.0]) - 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn test_maxsim() {
        // Query: 2 tokens, Doc: 3 tokens, dim=2
        let q1 = [1.0, 0.0];
        let q2 = [0.0, 1.0];
        let d1 = [0.5, 0.5];
        let d2 = [1.0, 0.0];
        let d3 = [0.0, 1.0];

        let query: Vec<&[f32]> = vec![&q1, &q2];
        let doc: Vec<&[f32]> = vec![&d1, &d2, &d3];

        // q1 best match: d2 (dot=1.0)
        // q2 best match: d3 (dot=1.0)
        // total: 2.0
        assert!((maxsim(&query, &doc) - 2.0).abs() < 1e-6);
    }
}

