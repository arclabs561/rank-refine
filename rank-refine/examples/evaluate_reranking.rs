//! Example: Evaluating reranking performance with rank-eval.
//!
//! This example shows how to use rank-eval to evaluate rank-refine's
//! reranking performance on a synthetic dataset.

use rank_eval::binary::Metrics;
use rank_eval::graded::{compute_ndcg, compute_map};
use rank_refine::colbert::rank;
use std::collections::{HashMap, HashSet};

fn main() {
    println!("=== Reranking Evaluation Example ===\n");

    // Simulate a query about "capital of France"
    let query = vec![
        vec![1.0, 0.0, 0.0], // "capital"
        vec![0.0, 1.0, 0.0], // "France"
    ];

    // Documents with varying relevance
    let docs = vec![
        (
            "doc1",
            vec![
                vec![0.95, 0.05, 0.0], // "Paris" (highly relevant)
                vec![0.05, 0.95, 0.0],  // "France"
            ],
        ),
        (
            "doc2",
            vec![
                vec![0.8, 0.2, 0.0],   // "city"
                vec![0.1, 0.9, 0.0],   // "France"
            ],
        ),
        (
            "doc3",
            vec![
                vec![0.2, 0.8, 0.0],   // "country"
                vec![0.0, 0.9, 0.1],   // "France"
            ],
        ),
        (
            "doc4",
            vec![
                vec![0.0, 0.0, 1.0],   // "unrelated"
                vec![0.0, 0.0, 1.0],   // "topic"
            ],
        ),
    ];

    println!("Query: 'capital of France'");
    println!("Documents: {} candidates\n", docs.len());

    // Rank documents using MaxSim
    let ranked = rank(&query, &docs);

    println!("Ranked Results:");
    for (i, (id, score)) in ranked.iter().enumerate() {
        println!("  {}. {} (score: {:.4})", i + 1, id, score);
    }
    println!();

    // Binary relevance evaluation
    let relevant: HashSet<_> = ["doc1", "doc2", "doc3"].into_iter().collect();
    let ranked_ids: Vec<&str> = ranked.iter().map(|(id, _)| *id).collect();

    let metrics = Metrics::compute(&ranked_ids, &relevant);

    println!("=== Binary Relevance Metrics ===");
    println!("Precision@1:  {:.4}", metrics.precision_at_1);
    println!("Precision@5:  {:.4}", metrics.precision_at_5);
    println!("Precision@10: {:.4}", metrics.precision_at_10);
    println!("Recall@5:     {:.4}", metrics.recall_at_5);
    println!("Recall@10:    {:.4}", metrics.recall_at_10);
    println!("MRR:          {:.4}", metrics.mrr);
    println!("nDCG@5:       {:.4}", metrics.ndcg_at_5);
    println!("nDCG@10:      {:.4}", metrics.ndcg_at_10);
    println!("MAP:          {:.4}", metrics.average_precision);
    println!();

    // Graded relevance evaluation
    let mut qrels = HashMap::new();
    qrels.insert("doc1".to_string(), 2); // Highly relevant
    qrels.insert("doc2".to_string(), 1); // Relevant
    qrels.insert("doc3".to_string(), 1); // Relevant
    qrels.insert("doc4".to_string(), 0); // Not relevant

    let ranked_with_scores: Vec<(String, f32)> = ranked
        .iter()
        .map(|(id, score)| (id.to_string(), *score))
        .collect();

    let ndcg_graded = compute_ndcg(&ranked_with_scores, &qrels, 10);
    let map_graded = compute_map(&ranked_with_scores, &qrels);

    println!("=== Graded Relevance Metrics ===");
    println!("nDCG@10 (graded): {:.4}", ndcg_graded);
    println!("MAP (graded):      {:.4}", map_graded);
    println!();

    // Analysis
    println!("=== Analysis ===");
    if metrics.ndcg_at_10 > 0.7 {
        println!("✓ Good ranking quality (nDCG@10 > 0.7)");
    } else {
        println!("⚠ Ranking quality could be improved");
    }

    if metrics.precision_at_1 > 0.5 {
        println!("✓ Top result is relevant");
    } else {
        println!("⚠ Top result may not be relevant");
    }
}

