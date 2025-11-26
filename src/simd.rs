//! Vector operations with SIMD acceleration.
//!
//! Provides `dot`, `cosine`, and `maxsim` with automatic SIMD dispatch:
//! - AVX2+FMA on `x86_64` (runtime detection)
//! - NEON on `aarch64`
//! - Portable fallback otherwise
//!
//! # Correctness
//!
//! All SIMD implementations are tested against the portable fallback
//! to ensure identical results (within floating-point tolerance).

/// Dot product of two vectors.
///
/// If vectors have different lengths, uses the shorter length.
/// Returns 0.0 for empty vectors.
#[inline]
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: We've verified AVX2 and FMA are available via runtime detection.
            // The function handles mismatched lengths by using min(a.len(), b.len()).
            return unsafe { dot_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is always available on aarch64.
        // The function handles mismatched lengths by using min(a.len(), b.len()).
        return unsafe { dot_neon(a, b) };
    }
    #[allow(unreachable_code)]
    dot_portable(a, b)
}

/// L2 norm of a vector.
#[inline]
#[must_use]
pub fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// Cosine similarity between two vectors.
///
/// Returns 0.0 if either vector has zero norm.
#[inline]
#[must_use]
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

/// `MaxSim`: sum over query tokens of max dot product with any doc token.
///
/// Used by ColBERT/PLAID for late interaction scoring.
///
/// Returns 0.0 if `query_tokens` is empty.
/// Returns 0.0 if `doc_tokens` is empty (no matches possible).
#[inline]
#[must_use]
pub fn maxsim(query_tokens: &[&[f32]], doc_tokens: &[&[f32]]) -> f32 {
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return 0.0;
    }
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

/// `MaxSim` with cosine similarity instead of dot product.
#[inline]
#[must_use]
pub fn maxsim_cosine(query_tokens: &[&[f32]], doc_tokens: &[&[f32]]) -> f32 {
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return 0.0;
    }
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

// ─────────────────────────────────────────────────────────────────────────────
// Convenience wrappers for owned vectors
// ─────────────────────────────────────────────────────────────────────────────

/// `MaxSim` for owned token vectors (convenience wrapper).
///
/// Equivalent to `maxsim(&as_slices(query), &as_slices(doc))` but more ergonomic.
///
/// # Example
///
/// ```rust
/// use rank_refine::simd::maxsim_vecs;
///
/// let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
/// let doc = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
/// let score = maxsim_vecs(&query, &doc);
/// ```
#[inline]
#[must_use]
pub fn maxsim_vecs(query_tokens: &[Vec<f32>], doc_tokens: &[Vec<f32>]) -> f32 {
    let q = crate::as_slices(query_tokens);
    let d = crate::as_slices(doc_tokens);
    maxsim(&q, &d)
}

/// `MaxSim` cosine for owned token vectors (convenience wrapper).
#[inline]
#[must_use]
pub fn maxsim_cosine_vecs(query_tokens: &[Vec<f32>], doc_tokens: &[Vec<f32>]) -> f32 {
    let q = crate::as_slices(query_tokens);
    let d = crate::as_slices(doc_tokens);
    maxsim_cosine(&q, &d)
}

// ─────────────────────────────────────────────────────────────────────────────
// Portable fallback
// ─────────────────────────────────────────────────────────────────────────────

/// Portable dot product implementation (reference for SIMD versions).
#[inline]
#[must_use]
pub(crate) fn dot_portable(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// AVX2 + FMA (x86_64)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::{
        __m256, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_setzero_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehl_ps, _mm_shuffle_ps,
    };

    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let chunks = n / 8;
    let remainder = n % 8;

    let mut sum: __m256 = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // SAFETY: We iterate chunks*8 elements, which is <= n <= min(a.len(), b.len()).
    // Pointer arithmetic stays within bounds.
    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum: reduce 8 f32s to 1
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);

    // Handle remainder with scalar ops
    let tail_start = chunks * 8;
    for i in 0..remainder {
        // SAFETY: tail_start + i < n, so within bounds
        result += *a.get_unchecked(tail_start + i) * *b.get_unchecked(tail_start + i);
    }

    result
}

// ─────────────────────────────────────────────────────────────────────────────
// NEON (aarch64)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::{float32x4_t, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum: float32x4_t = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    // SAFETY: We iterate chunks*4 elements, which is <= n <= min(a.len(), b.len()).
    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        sum = vfmaq_f32(sum, va, vb);
    }

    // Horizontal sum: reduce 4 f32s to 1
    let mut result = vaddvq_f32(sum);

    // Handle remainder with scalar ops
    let tail_start = chunks * 4;
    for i in 0..remainder {
        // SAFETY: tail_start + i < n, so within bounds
        result += *a.get_unchecked(tail_start + i) * *b.get_unchecked(tail_start + i);
    }

    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_basic() {
        assert!((dot(&[1.0, 2.0], &[3.0, 4.0]) - 11.0).abs() < 1e-5);
    }

    #[test]
    fn test_dot_empty() {
        assert_eq!(dot(&[], &[]), 0.0);
        assert_eq!(dot(&[1.0], &[]), 0.0);
        assert_eq!(dot(&[], &[1.0]), 0.0);
    }

    #[test]
    fn test_dot_mismatched_lengths() {
        // Should use shorter length
        assert!((dot(&[1.0, 2.0, 3.0], &[4.0, 5.0]) - 14.0).abs() < 1e-5); // 1*4 + 2*5 = 14
    }

    #[test]
    fn test_dot_simd_vs_portable() {
        // Test various lengths around SIMD boundaries
        for len in [
            0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 256, 1024,
        ] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32) * 0.2 + 1.0).collect();

            let portable = dot_portable(&a, &b);
            let simd = dot(&a, &b);

            // Use relative tolerance for larger values
            let tolerance = (portable.abs() * 1e-5).max(1e-5);
            assert!(
                (portable - simd).abs() < tolerance,
                "Mismatch at len={}: portable={}, simd={}, diff={}",
                len,
                portable,
                simd,
                (portable - simd).abs()
            );
        }
    }

    #[test]
    fn test_cosine_basic() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-5);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_zero_norm() {
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
        assert_eq!(cosine(&[1.0, 0.0], &[0.0, 0.0]), 0.0);
    }

    #[test]
    fn test_maxsim_basic() {
        let q1 = [1.0, 0.0];
        let q2 = [0.0, 1.0];
        let d1 = [0.5, 0.5];
        let d2 = [1.0, 0.0];
        let d3 = [0.0, 1.0];

        let query: Vec<&[f32]> = vec![&q1, &q2];
        let doc: Vec<&[f32]> = vec![&d1, &d2, &d3];

        // q1's best match is d2 (dot=1.0), q2's best match is d3 (dot=1.0)
        assert!((maxsim(&query, &doc) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_maxsim_empty_query() {
        let doc: Vec<&[f32]> = vec![&[1.0, 0.0]];
        assert_eq!(maxsim(&[], &doc), 0.0);
    }

    #[test]
    fn test_maxsim_empty_doc() {
        let q1 = [1.0, 0.0];
        let query: Vec<&[f32]> = vec![&q1];
        // With empty docs, returns 0.0 (no matches possible)
        assert_eq!(maxsim(&query, &[]), 0.0);
    }

    #[test]
    fn test_maxsim_vecs() {
        let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let doc = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        // Both query tokens find perfect matches
        assert!((maxsim_vecs(&query, &doc) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_maxsim_cosine_vecs() {
        let query = vec![vec![2.0, 0.0]]; // unnormalized
        let doc = vec![vec![1.0, 0.0]];
        // Cosine should normalize, so result is 1.0
        assert!((maxsim_cosine_vecs(&query, &doc) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_maxsim_cosine_basic() {
        let q1 = [1.0, 0.0];
        let d1 = [1.0, 0.0];
        let d2 = [0.0, 1.0];

        let query: Vec<&[f32]> = vec![&q1];
        let doc: Vec<&[f32]> = vec![&d1, &d2];

        // q1's best cosine match is d1 (cosine=1.0)
        assert!((maxsim_cosine(&query, &doc) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_maxsim_cosine_empty_doc() {
        let q1 = [1.0, 0.0];
        let query: Vec<&[f32]> = vec![&q1];
        assert_eq!(maxsim_cosine(&query, &[]), 0.0);
    }

    // ───────────────────────────────────────────────────────────────────────
    // Mutation-killing tests: verify exact mathematical properties
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn cosine_zero_norm_returns_zero_not_nan() {
        // When either vector has zero norm, cosine returns 0.0
        // This tests the `> 0.0` check - changing to `>= 0.0` would cause NaN
        let zero = [0.0, 0.0];
        let nonzero = [1.0, 2.0];

        let c1 = cosine(&zero, &nonzero);
        let c2 = cosine(&nonzero, &zero);
        let c3 = cosine(&zero, &zero);

        assert_eq!(c1, 0.0, "cosine(zero, x) should be 0, got {}", c1);
        assert_eq!(c2, 0.0, "cosine(x, zero) should be 0, got {}", c2);
        assert_eq!(c3, 0.0, "cosine(zero, zero) should be 0, got {}", c3);
        assert!(!c1.is_nan(), "should not return NaN");
    }

    #[test]
    fn cosine_near_zero_norm_stable() {
        // Very small norms below threshold return 0.0 for stability
        let tiny = [1e-20, 0.0];
        let normal = [1.0, 0.0];

        let c = cosine(&tiny, &normal);
        assert!(c.is_finite(), "cosine with tiny norm should be finite");
        // Returns 0.0 for stability when norm < 1e-9
        assert_eq!(c, 0.0, "tiny norm should return 0.0");

        // Small but above threshold should work
        let small = [1e-8, 0.0];
        let c2 = cosine(&small, &normal);
        assert!(c2.is_finite());
        assert!(
            (c2 - 1.0).abs() < 1e-3,
            "parallel vectors above threshold should have cosine ~1"
        );
    }

    #[test]
    fn dot_exact_orthogonal() {
        // Orthogonal vectors have dot product 0
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        assert_eq!(dot(&a, &b), 0.0);
    }

    #[test]
    fn dot_exact_parallel() {
        // Parallel unit vectors have dot product 1
        let a = [1.0, 0.0];
        let b = [1.0, 0.0];
        assert!((dot(&a, &b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn maxsim_single_query_single_doc() {
        // Simplest case: 1 query token, 1 doc token
        let q = [1.0, 2.0, 3.0];
        let d = [4.0, 5.0, 6.0];

        let query: Vec<&[f32]> = vec![&q];
        let doc: Vec<&[f32]> = vec![&d];

        let expected_dot = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0; // 4 + 10 + 18 = 32
        let actual = maxsim(&query, &doc);

        assert!(
            (actual - expected_dot).abs() < 1e-5,
            "expected {}, got {}",
            expected_dot,
            actual
        );
    }

    #[test]
    fn maxsim_sum_of_maxes() {
        // MaxSim = sum over query tokens of max(dot with each doc token)
        let q1 = [1.0, 0.0];
        let q2 = [0.0, 1.0];
        let d1 = [0.5, 0.0]; // dot(q1,d1)=0.5, dot(q2,d1)=0.0
        let d2 = [0.0, 0.8]; // dot(q1,d2)=0.0, dot(q2,d2)=0.8

        let query: Vec<&[f32]> = vec![&q1, &q2];
        let doc: Vec<&[f32]> = vec![&d1, &d2];

        // max for q1 is 0.5 (from d1), max for q2 is 0.8 (from d2)
        let expected = 0.5 + 0.8;
        let actual = maxsim(&query, &doc);

        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {}, got {}",
            expected,
            actual
        );
    }

    #[test]
    fn norm_exact_values() {
        // Test exact norm calculations
        assert!((norm(&[3.0, 4.0]) - 5.0).abs() < 1e-9, "3-4-5 triangle");
        assert!((norm(&[1.0, 0.0]) - 1.0).abs() < 1e-9, "unit x");
        assert!((norm(&[0.0, 0.0]) - 0.0).abs() < 1e-9, "zero vector");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Property Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_vec(len: usize) -> impl Strategy<Value = Vec<f32>> {
        proptest::collection::vec(-10.0f32..10.0, len)
    }

    proptest! {
        /// SIMD dot matches portable implementation
        #[test]
        fn dot_simd_matches_portable(a in arb_vec(128), b in arb_vec(128)) {
            let simd_result = dot(&a, &b);
            let portable_result = dot_portable(&a, &b);
            prop_assert!(
                (simd_result - portable_result).abs() < 1e-3,
                "SIMD {} != portable {}",
                simd_result,
                portable_result
            );
        }

        /// Dot product is commutative: dot(a, b) == dot(b, a)
        #[test]
        fn dot_commutative(a in arb_vec(64), b in arb_vec(64)) {
            let ab = dot(&a, &b);
            let ba = dot(&b, &a);
            prop_assert!((ab - ba).abs() < 1e-5);
        }

        /// Cosine similarity is in [-1, 1] for non-zero vectors
        #[test]
        fn cosine_bounded(
            a in arb_vec(32).prop_filter("non-zero", |v| v.iter().any(|x| x.abs() > 1e-6)),
            b in arb_vec(32).prop_filter("non-zero", |v| v.iter().any(|x| x.abs() > 1e-6))
        ) {
            let c = cosine(&a, &b);
            prop_assert!(c >= -1.0 - 1e-5 && c <= 1.0 + 1e-5, "cosine {} out of bounds", c);
        }

        /// Cosine similarity is commutative
        #[test]
        fn cosine_commutative(a in arb_vec(32), b in arb_vec(32)) {
            let ab = cosine(&a, &b);
            let ba = cosine(&b, &a);
            prop_assert!((ab - ba).abs() < 1e-5);
        }

        /// MaxSim is non-negative when all dot products are non-negative
        #[test]
        fn maxsim_nonnegative_inputs(
            q_data in proptest::collection::vec(arb_vec(16), 1..5),
            d_data in proptest::collection::vec(arb_vec(16), 1..5)
        ) {
            // Make all vectors have non-negative components
            let q_pos: Vec<Vec<f32>> = q_data.iter()
                .map(|v| v.iter().map(|x| x.abs()).collect())
                .collect();
            let d_pos: Vec<Vec<f32>> = d_data.iter()
                .map(|v| v.iter().map(|x| x.abs()).collect())
                .collect();

            let q_refs: Vec<&[f32]> = q_pos.iter().map(|v| v.as_slice()).collect();
            let d_refs: Vec<&[f32]> = d_pos.iter().map(|v| v.as_slice()).collect();

            let score = maxsim(&q_refs, &d_refs);
            prop_assert!(score >= 0.0, "maxsim {} should be non-negative", score);
        }

        /// Empty inputs return 0
        #[test]
        fn maxsim_empty_returns_zero(q_data in proptest::collection::vec(arb_vec(8), 0..3)) {
            let q_refs: Vec<&[f32]> = q_data.iter().map(|v| v.as_slice()).collect();
            let empty: Vec<&[f32]> = vec![];

            // Empty query always returns 0
            prop_assert_eq!(maxsim(&empty, &q_refs), 0.0);
            // Empty doc always returns 0
            prop_assert_eq!(maxsim(&q_refs, &empty), 0.0);
        }

        // ─────────────────────────────────────────────────────────────────────────
        // Additional mathematical properties
        // ─────────────────────────────────────────────────────────────────────────

        /// Dot product with self equals squared L2 norm
        #[test]
        fn dot_self_is_squared_norm(v in arb_vec(32)) {
            let dot_self = dot(&v, &v);
            let n = norm(&v);
            let squared_norm = n * n;
            // Use relative tolerance for large values
            let tolerance = (squared_norm.abs() * 1e-4).max(1e-4);
            prop_assert!(
                (dot_self - squared_norm).abs() < tolerance,
                "dot(v,v) = {} but norm²= {}",
                dot_self,
                squared_norm
            );
        }

        /// Cosine with self is 1 (for non-zero vectors)
        #[test]
        fn cosine_self_is_one(v in arb_vec(16).prop_filter("non-zero", |v| norm(v) > 1e-6)) {
            let c = cosine(&v, &v);
            prop_assert!(
                (c - 1.0).abs() < 1e-5,
                "cosine(v, v) = {} should be 1",
                c
            );
        }

        /// Norm is non-negative
        #[test]
        fn norm_nonnegative(v in arb_vec(64)) {
            let n = norm(&v);
            prop_assert!(n >= 0.0, "norm {} should be non-negative", n);
        }

        /// Norm of scaled vector: ||αv|| = |α| ||v||
        #[test]
        fn norm_scaling(v in arb_vec(16), alpha in -10.0f32..10.0) {
            let scaled: Vec<f32> = v.iter().map(|x| x * alpha).collect();
            let n_v = norm(&v);
            let n_scaled = norm(&scaled);
            let expected = alpha.abs() * n_v;
            prop_assert!(
                (n_scaled - expected).abs() < 1e-4,
                "||αv|| = {} but |α|||v|| = {}",
                n_scaled,
                expected
            );
        }

        /// Dot product is bilinear: dot(αa, b) = α·dot(a, b)
        #[test]
        fn dot_bilinear(a in arb_vec(16), b in arb_vec(16), alpha in -5.0f32..5.0) {
            let scaled_a: Vec<f32> = a.iter().map(|x| x * alpha).collect();
            let dot_scaled = dot(&scaled_a, &b);
            let expected = alpha * dot(&a, &b);
            prop_assert!(
                (dot_scaled - expected).abs() < 1e-3,
                "dot(αa, b) = {} but α·dot(a, b) = {}",
                dot_scaled,
                expected
            );
        }

        /// Cauchy-Schwarz: |dot(a, b)| <= ||a|| ||b||
        #[test]
        fn cauchy_schwarz(a in arb_vec(32), b in arb_vec(32)) {
            let d = dot(&a, &b).abs();
            let bound = norm(&a) * norm(&b);
            prop_assert!(
                d <= bound + 1e-4,
                "|dot(a,b)| = {} should be <= ||a||·||b|| = {}",
                d,
                bound
            );
        }

        /// MaxSim scales linearly with query token count for identical matches
        #[test]
        fn maxsim_scales_with_query_count(n_query in 1usize..5, dim in 4usize..8) {
            // Create identical query and doc token
            let token: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
            let query: Vec<Vec<f32>> = vec![token.clone(); n_query];
            let doc = vec![token.clone()];

            let q_refs: Vec<&[f32]> = query.iter().map(Vec::as_slice).collect();
            let d_refs: Vec<&[f32]> = doc.iter().map(Vec::as_slice).collect();

            let score = maxsim(&q_refs, &d_refs);
            // Each query token has max sim = ||token||² (dot with itself)
            let expected = n_query as f32 * dot(&token, &token);

            prop_assert!(
                (score - expected).abs() < 1e-4,
                "MaxSim should scale linearly: {} vs expected {}",
                score,
                expected
            );
        }

        /// MaxSim with normalized vectors bounded by query count
        #[test]
        fn maxsim_cosine_bounded_by_query_count(n_query in 1usize..4, n_doc in 1usize..4) {
            // Create unit vectors
            let query: Vec<Vec<f32>> = (0..n_query)
                .map(|i| {
                    let mut v = vec![0.0f32; 8];
                    v[i % 8] = 1.0;
                    v
                })
                .collect();
            let doc: Vec<Vec<f32>> = (0..n_doc)
                .map(|i| {
                    let mut v = vec![0.0f32; 8];
                    v[(i + 1) % 8] = 1.0;
                    v
                })
                .collect();

            let q_refs: Vec<&[f32]> = query.iter().map(Vec::as_slice).collect();
            let d_refs: Vec<&[f32]> = doc.iter().map(Vec::as_slice).collect();

            let score = maxsim_cosine(&q_refs, &d_refs);

            // Each query token contributes at most 1 (max cosine with any doc token)
            let upper_bound = n_query as f32;
            prop_assert!(
                score <= upper_bound + 1e-5,
                "MaxSim cosine {} should be <= {}",
                score,
                upper_bound
            );
        }
    }
}
