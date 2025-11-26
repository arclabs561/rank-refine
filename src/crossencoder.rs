//! Cross-encoder reranking.
//!
//! Cross-encoders score query-document pairs directly with a transformer,
//! providing high precision at the cost of O(n) inference calls.
//!
//! ## Usage
//!
//! This module provides a trait-based API. Implement [`CrossEncoderModel`] for
//! your inference backend (e.g., candle, ort, tch).
//!
//! ```rust
//! use rank_refine::crossencoder::{CrossEncoderModel, rerank};
//!
//! struct MyModel;
//!
//! impl CrossEncoderModel for MyModel {
//!     fn score_batch(&self, query: &str, documents: &[&str]) -> Vec<f32> {
//!         // Your inference code here
//!         documents.iter().map(|d| d.len() as f32 / 100.0).collect()
//!     }
//! }
//!
//! let model = MyModel;
//! let candidates = vec![("doc1", "short"), ("doc2", "longer text here")];
//! let ranked = rerank(&model, "query", &candidates);
//! assert_eq!(ranked[0].0, "doc2"); // longer doc scores higher
//! ```

/// A query-document pair score.
pub type Score = f32;

/// Trait for cross-encoder implementations.
///
/// Implementors score query-document pairs using transformer models.
pub trait CrossEncoderModel {
    /// Score a batch of query-document pairs.
    ///
    /// Returns relevance scores (higher = more relevant).
    fn score_batch(&self, query: &str, documents: &[&str]) -> Vec<Score>;

    /// Score a single query-document pair.
    fn score(&self, query: &str, document: &str) -> Score {
        self.score_batch(query, &[document]).pop().unwrap_or(0.0)
    }
}

/// Rerank candidates using a cross-encoder.
///
/// # Arguments
///
/// * `model` - Cross-encoder model
/// * `query` - Query string
/// * `candidates` - Document id and text pairs
///
/// # Returns
///
/// Candidates sorted by descending cross-encoder score.
#[must_use]
pub fn rerank<I: Clone, M: CrossEncoderModel>(
    model: &M,
    query: &str,
    candidates: &[(I, &str)],
) -> Vec<(I, Score)> {
    let docs: Vec<&str> = candidates.iter().map(|(_, text)| *text).collect();
    let scores = model.score_batch(query, &docs);

    let mut results: Vec<(I, Score)> = candidates
        .iter()
        .zip(scores)
        .map(|((id, _), score)| (id.clone(), score))
        .collect();

    // Use total_cmp for deterministic sorting (handles NaN)
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
    results
}

/// Blend cross-encoder scores with original retrieval scores.
///
/// Cross-encoder scores are normalized to [0, 1] before blending.
///
/// # Arguments
///
/// * `model` - Cross-encoder model
/// * `query` - Query string
/// * `candidates` - Tuples of (id, text, original score)
/// * `alpha` - Weight for original score (0.0 = all CE, 1.0 = all original)
#[must_use]
pub fn refine<I: Clone, M: CrossEncoderModel>(
    model: &M,
    query: &str,
    candidates: &[(I, &str, f32)], // (id, text, original_score)
    alpha: f32,
) -> Vec<(I, Score)> {
    if candidates.is_empty() {
        return Vec::new();
    }

    let docs: Vec<&str> = candidates.iter().map(|(_, text, _)| *text).collect();
    let ce_scores = model.score_batch(query, &docs);

    // Normalize CE scores to [0, 1]
    let (min_ce, max_ce) = ce_scores
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &s| {
            (lo.min(s), hi.max(s))
        });
    let range = (max_ce - min_ce).max(1e-9);

    let mut results: Vec<(I, Score)> = candidates
        .iter()
        .zip(ce_scores)
        .map(|((id, _, orig), ce)| {
            let ce_norm = (ce - min_ce) / range;
            // Use mul_add for better precision
            let blended = (1.0 - alpha).mul_add(ce_norm, alpha * orig);
            (id.clone(), blended)
        })
        .collect();

    // Use total_cmp for deterministic sorting (handles NaN)
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockEncoder;

    impl CrossEncoderModel for MockEncoder {
        fn score_batch(&self, _query: &str, documents: &[&str]) -> Vec<Score> {
            // Score by document length (longer = higher)
            documents.iter().map(|d| d.len() as f32 / 100.0).collect()
        }
    }

    #[test]
    fn test_rerank() {
        let model = MockEncoder;
        let candidates = vec![
            ("d1", "short"),
            ("d2", "this is a longer document"),
            ("d3", "medium length"),
        ];

        let ranked = rerank(&model, "query", &candidates);
        assert_eq!(ranked[0].0, "d2"); // longest
    }

    #[test]
    fn test_rerank_empty() {
        let model = MockEncoder;
        let candidates: Vec<(&str, &str)> = vec![];

        let ranked = rerank(&model, "query", &candidates);
        assert!(ranked.is_empty());
    }

    #[test]
    fn test_refine() {
        let model = MockEncoder;
        let candidates = vec![
            ("d1", "short", 0.9),           // high original, low CE
            ("d2", "longer doc here", 0.1), // low original, high CE
        ];

        // With alpha=0 (all CE), d2 wins
        let refined = refine(&model, "query", &candidates, 0.0);
        assert_eq!(refined[0].0, "d2");

        // With alpha=1 (all original), d1 wins
        let refined = refine(&model, "query", &candidates, 1.0);
        assert_eq!(refined[0].0, "d1");
    }

    #[test]
    fn test_refine_empty() {
        let model = MockEncoder;
        let candidates: Vec<(&str, &str, f32)> = vec![];

        let refined = refine(&model, "query", &candidates, 0.5);
        assert!(refined.is_empty());
    }

    #[test]
    fn test_refine_single_candidate() {
        let model = MockEncoder;
        let candidates = vec![("d1", "some text", 0.5)];

        // With single candidate, CE normalization gives 0 (or 1), should not panic
        let refined = refine(&model, "query", &candidates, 0.5);
        assert_eq!(refined.len(), 1);
        assert_eq!(refined[0].0, "d1");
    }

    #[test]
    fn test_nan_score_handling() {
        let model = MockEncoder;
        let candidates = vec![
            ("d1", "text", f32::NAN),
            ("d2", "text", 0.5),
        ];

        // Should not panic; total_cmp puts NaN > all numbers, so NaN comes first in descending sort
        let refined = refine(&model, "query", &candidates, 0.5);
        assert_eq!(refined.len(), 2);
        assert!(refined[0].1.is_nan());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    struct LengthEncoder;

    impl CrossEncoderModel for LengthEncoder {
        fn score_batch(&self, _query: &str, documents: &[&str]) -> Vec<Score> {
            documents.iter().map(|d| d.len() as f32).collect()
        }
    }

    proptest! {
        /// Rerank output length equals input length
        #[test]
        fn rerank_preserves_count(n in 0usize..10) {
            let model = LengthEncoder;
            let candidates: Vec<(u32, &str)> = (0..n as u32)
                .map(|i| (i, "test"))
                .collect();

            let ranked = rerank(&model, "query", &candidates);
            prop_assert_eq!(ranked.len(), n);
        }

        /// Rerank results are sorted by descending score
        #[test]
        fn rerank_sorted_descending(texts in proptest::collection::vec("[a-z]{1,20}", 2..8)) {
            let model = LengthEncoder;
            let candidates: Vec<(usize, &str)> = texts.iter()
                .enumerate()
                .map(|(i, s)| (i, s.as_str()))
                .collect();

            let ranked = rerank(&model, "query", &candidates);
            for window in ranked.windows(2) {
                prop_assert!(
                    window[0].1 >= window[1].1,
                    "Results not sorted: {:?}",
                    ranked
                );
            }
        }

        /// Refine output length equals input length
        #[test]
        fn refine_preserves_count(n in 0usize..10) {
            let model = LengthEncoder;
            let candidates: Vec<(u32, &str, f32)> = (0..n as u32)
                .map(|i| (i, "test", i as f32 * 0.1))
                .collect();

            let refined = refine(&model, "query", &candidates, 0.5);
            prop_assert_eq!(refined.len(), n);
        }

        /// Alpha=1.0 preserves original score ordering
        #[test]
        fn alpha_one_preserves_original_order(scores in proptest::collection::vec(0.0f32..1.0, 3..6)) {
            let model = LengthEncoder;
            // All same text so CE scores are equal, only original scores matter
            let candidates: Vec<(usize, &str, f32)> = scores.iter()
                .enumerate()
                .map(|(i, &s)| (i, "same", s))
                .collect();

            let refined = refine(&model, "query", &candidates, 1.0);

            // With alpha=1, order should match original scores descending
            let mut expected_order: Vec<(usize, f32)> = scores.iter()
                .enumerate()
                .map(|(i, &s)| (i, s))
                .collect();
            expected_order.sort_by(|a, b| b.1.total_cmp(&a.1));

            for (i, (id, _)) in refined.iter().enumerate() {
                prop_assert_eq!(*id, expected_order[i].0);
            }
        }

        /// Blended scores are bounded when inputs are bounded
        #[test]
        fn refine_scores_bounded(
            orig_scores in proptest::collection::vec(0.0f32..1.0, 2..5),
            alpha in 0.0f32..=1.0
        ) {
            let model = LengthEncoder;
            let candidates: Vec<(usize, &str, f32)> = orig_scores.iter()
                .enumerate()
                .map(|(i, &s)| (i, "text", s))
                .collect();

            let refined = refine(&model, "query", &candidates, alpha);

            // CE scores are normalized to [0,1], originals in [0,1], so blended should be in [0,1]
            for (_, score) in &refined {
                prop_assert!(
                    *score >= -0.01 && *score <= 1.01,
                    "Score {} out of expected bounds",
                    score
                );
            }
        }
    }
}
