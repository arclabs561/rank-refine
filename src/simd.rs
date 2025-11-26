//! Vector operations with SIMD acceleration.
//!
//! Provides `dot`, `cosine`, and `maxsim` with automatic SIMD dispatch:
//! - AVX2 on x86_64 (when available at runtime)
//! - NEON on aarch64 (always available)
//! - Portable fallback otherwise

/// Dot product of two vectors.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
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

/// MaxSim with cosine similarity.
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

// ─────────────────────────────────────────────────────────────────────────────
// Portable fallback
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn dot_portable(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// AVX2 + FMA (x86_64)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let n = a.len().min(b.len());
    let chunks = n / 8;
    let remainder = n % 8;

    let mut sum = _mm256_setzero_ps();

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);

    // Handle remainder
    let tail_start = chunks * 8;
    for i in 0..remainder {
        result += a[tail_start + i] * b[tail_start + i];
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
    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum = vdupq_n_f32(0.0);

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        sum = vfmaq_f32(sum, va, vb);
    }

    // Horizontal sum
    let mut result = vaddvq_f32(sum);

    // Handle remainder
    let tail_start = chunks * 4;
    for i in 0..remainder {
        result += a[tail_start + i] * b[tail_start + i];
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot() {
        assert!((dot(&[1.0, 2.0], &[3.0, 4.0]) - 11.0).abs() < 1e-5);

        // Larger vector
        let a: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..256).map(|i| (i * 2) as f32).collect();
        let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot(&a, &b) - expected).abs() < 1.0);
    }

    #[test]
    fn test_cosine() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-5);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-5);
    }

    #[test]
    fn test_maxsim() {
        let q1 = [1.0, 0.0];
        let q2 = [0.0, 1.0];
        let d1 = [0.5, 0.5];
        let d2 = [1.0, 0.0];
        let d3 = [0.0, 1.0];

        let query: Vec<&[f32]> = vec![&q1, &q2];
        let doc: Vec<&[f32]> = vec![&d1, &d2, &d3];

        assert!((maxsim(&query, &doc) - 2.0).abs() < 1e-5);
    }
}
