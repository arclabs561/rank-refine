//! Comprehensive benchmarks for all rank-refine algorithms.
//!
//! Run: `cargo bench --bench comprehensive`

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rank_refine::*;

fn generate_token_embeddings(n_tokens: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n_tokens)
        .map(|i| {
            (0..dim)
                .map(|j| ((i * 7 + j * 11) % 100) as f32 / 100.0 - 0.5)
                .collect()
        })
        .collect()
}

fn generate_documents(n_docs: usize, tokens_per_doc: usize, dim: usize) -> Vec<(String, Vec<Vec<f32>>)> {
    (0..n_docs)
        .map(|i| {
            let doc_id = format!("doc_{}", i);
            let tokens = generate_token_embeddings(tokens_per_doc, dim);
            (doc_id, tokens)
        })
        .collect()
}

fn bench_maxsim(c: &mut Criterion) {
    let mut group = c.benchmark_group("maxsim");
    
    for (n_query, n_doc, dim) in [
        (10, 50, 128),
        (20, 100, 128),
        (30, 200, 128),
        (10, 50, 768),
    ].iter() {
        let query = generate_token_embeddings(*n_query, *dim);
        let doc = generate_token_embeddings(*n_doc, *dim);
        
        let query_slices: Vec<&[f32]> = query.iter().map(|v| v.as_slice()).collect();
        let doc_slices: Vec<&[f32]> = doc.iter().map(|v| v.as_slice()).collect();
        
        group.bench_with_input(
            BenchmarkId::new("maxsim", format!("q{}_d{}_dim{}", n_query, n_doc, dim)),
            &(query_slices, doc_slices),
            |b, (q, d)| {
                b.iter(|| maxsim(black_box(q), black_box(d)))
            },
        );
    }
    
    group.finish();
}

fn bench_cosine_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_similarity");
    
    for dim in [128, 256, 384, 768].iter() {
        let vec1: Vec<f32> = (0..*dim).map(|i| (i % 100) as f32 / 100.0).collect();
        let vec2: Vec<f32> = (0..*dim).map(|i| ((i + 1) % 100) as f32 / 100.0).collect();
        
        group.bench_with_input(
            BenchmarkId::new("cosine", dim),
            dim,
            |b, _| {
                b.iter(|| cosine(black_box(&vec1), black_box(&vec2)))
            },
        );
    }
    
    group.finish();
}

fn bench_token_pooling(c: &mut Criterion) {
    let mut group = c.benchmark_group("token_pooling");
    
    for (n_tokens, pool_factor) in [
        (50, 2),
        (100, 2),
        (200, 2),
        (100, 3),
        (100, 4),
    ].iter() {
        let tokens = generate_token_embeddings(*n_tokens, 128);
        
        group.bench_with_input(
            BenchmarkId::new("pool_tokens", format!("{}_factor{}", n_tokens, pool_factor)),
            &tokens,
            |b, tokens| {
                b.iter(|| {
                    colbert::pool_tokens(black_box(tokens), black_box(*pool_factor))
                        .unwrap()
                })
            },
        );
    }
    
    group.finish();
}

fn bench_mmr(c: &mut Criterion) {
    let mut group = c.benchmark_group("mmr");
    
    for (n_candidates, k) in [
        (50, 10),
        (100, 20),
        (200, 50),
        (500, 100),
    ].iter() {
        let candidates: Vec<(&str, f32)> = (0..*n_candidates)
            .map(|i| (format!("doc_{}", i).as_str(), (n_candidates - i) as f32))
            .collect();
        
        // Generate similarity matrix (flattened, row-major)
        let mut similarity = Vec::new();
        for i in 0..*n_candidates {
            for j in 0..*n_candidates {
                if i == j {
                    similarity.push(1.0);
                } else {
                    let sim = 1.0 - ((i as f32 - j as f32).abs() / *n_candidates as f32);
                    similarity.push(sim.max(0.0));
                }
            }
        }
        
        let config = diversity::MmrConfig::default().with_k(*k);
        
        group.bench_with_input(
            BenchmarkId::new("mmr", format!("{}_k{}", n_candidates, k)),
            &(candidates, similarity, config),
            |b, (cands, sim, cfg)| {
                b.iter(|| mmr(black_box(cands), black_box(sim), black_box(*cfg)))
            },
        );
    }
    
    group.finish();
}

fn bench_colbert_rank(c: &mut Criterion) {
    let mut group = c.benchmark_group("colbert_rank");
    
    for (n_docs, tokens_per_doc) in [
        (10, 50),
        (50, 100),
        (100, 100),
    ].iter() {
        let query = generate_token_embeddings(10, 128);
        let docs = generate_documents(*n_docs, *tokens_per_doc, 128);
        
        group.bench_with_input(
            BenchmarkId::new("colbert_rank", format!("{}docs_{}tokens", n_docs, tokens_per_doc)),
            &(query, docs),
            |b, (q, d)| {
                b.iter(|| colbert::rank(black_box(q), black_box(d)))
            },
        );
    }
    
    group.finish();
}

fn bench_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalization");
    
    for dim in [128, 256, 384, 768, 1536].iter() {
        let mut vec: Vec<f32> = (0..*dim).map(|i| (i % 100) as f32 / 10.0).collect();
        
        group.bench_with_input(
            BenchmarkId::new("normalize", dim),
            &mut vec,
            |b, v| {
                b.iter(|| {
                    let mut v_clone = v.clone();
                    embedding::normalize(black_box(&mut v_clone)).unwrap();
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_maxsim,
    bench_cosine_similarity,
    bench_token_pooling,
    bench_mmr,
    bench_colbert_rank,
    bench_normalization
);
criterion_main!(benches);

