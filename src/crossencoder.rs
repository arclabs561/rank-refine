//! Cross-encoder reranking (stub).
//!
//! Cross-encoders score query-document pairs directly with a transformer,
//! providing high precision at the cost of O(n) inference calls.
//!
//! ## Status
//!
//! This module defines the API. Actual inference requires the `candle` feature.
//!
//! ## Planned Usage
//!
//! ```rust,ignore
//! use rank_refine::crossencoder::{CrossEncoder, Config};
//!
//! let model = CrossEncoder::from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")?;
//! let candidates = vec![("doc1", "Paris is the capital of France.")];
//! let scores = model.score("What is the capital of France?", &candidates);
//! ```

/// Cross-encoder configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// Maximum sequence length (query + doc + special tokens).
    pub max_length: usize,
    /// Batch size for inference.
    pub batch_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_length: 512,
            batch_size: 32,
        }
    }
}

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

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Blend cross-encoder scores with original retrieval scores.
pub fn refine<I: Clone, M: CrossEncoderModel>(
    model: &M,
    query: &str,
    candidates: &[(I, &str, f32)], // (id, text, original_score)
    alpha: f32,                    // weight for original score
) -> Vec<(I, Score)> {
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
            let blended = alpha * orig + (1.0 - alpha) * ce_norm;
            (id.clone(), blended)
        })
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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
}
