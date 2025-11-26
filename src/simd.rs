//! Vector operations with SIMD acceleration.
//!
//! Provides `dot`, `cosine`, and `maxsim` with automatic SIMD dispatch:
//! - AVX2+FMA on x86_64 (runtime detection)
//! - NEON on aarch64
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
pub fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// Cosine similarity between two vectors.
///
/// Returns 0.0 if either vector has zero norm.
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

/// MaxSim: sum over query tokens of max dot product with any doc token.
///
/// Used by ColBERT/PLAID for late interaction scoring.
///
/// Returns 0.0 if `query_tokens` is empty.
/// Returns `NEG_INFINITY` summed for each query token if `doc_tokens` is empty
/// (this is likely a bug in usage — consider checking inputs).
#[inline]
pub fn maxsim(query_tokens: &[&[f32]], doc_tokens: &[&[f32]]) -> f32 {
    if query_tokens.is_empty() {
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

/// MaxSim with cosine similarity instead of dot product.
#[inline]
pub fn maxsim_cosine(query_tokens: &[&[f32]], doc_tokens: &[&[f32]]) -> f32 {
    if query_tokens.is_empty() {
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
// Portable fallback
// ─────────────────────────────────────────────────────────────────────────────

/// Portable dot product implementation.
///
/// This is the reference implementation that SIMD versions must match.
#[inline]
pub fn dot_portable(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// AVX2 + FMA (x86_64)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let chunks = n / 8;
    let remainder = n % 8;

    let mut sum = _mm256_setzero_ps();

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
    use std::arch::aarch64::*;

    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum = vdupq_n_f32(0.0);

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
        for len in [0, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 100, 256] {
            let a: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32) * 0.2 + 1.0).collect();

            let portable = dot_portable(&a, &b);
            let simd = dot(&a, &b);

            assert!(
                (portable - simd).abs() < 1e-3,
                "Mismatch at len={}: portable={}, simd={}",
                len,
                portable,
                simd
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
        // With empty docs, each query token contributes NEG_INFINITY
        assert!(maxsim(&query, &[]).is_infinite() && maxsim(&query, &[]).is_sign_negative());
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
}
