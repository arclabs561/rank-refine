use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rank_refine::{colbert, diversity, matryoshka, simd};

fn random_vec(dim: usize, seed: u64) -> Vec<f32> {
    // Simple LCG for reproducible "random" vectors
    let mut x = seed;
    (0..dim)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            (x as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn bench_simd(c: &mut Criterion) {
    let mut g = c.benchmark_group("simd");

    for &dim in &[128, 384, 768, 1536] {
        let a = random_vec(dim, 1);
        let b = random_vec(dim, 2);

        g.bench_with_input(BenchmarkId::new("dot", dim), &dim, |bench, _| {
            bench.iter(|| black_box(simd::dot(&a, &b)));
        });

        g.bench_with_input(BenchmarkId::new("cosine", dim), &dim, |bench, _| {
            bench.iter(|| black_box(simd::cosine(&a, &b)));
        });
    }

    g.finish();
}

fn bench_maxsim(c: &mut Criterion) {
    let mut g = c.benchmark_group("maxsim");

    // Typical ColBERT: 32 query tokens, 128 doc tokens, 128 dims
    let dim = 128;
    let query: Vec<Vec<f32>> = (0..32).map(|i| random_vec(dim, i)).collect();
    let doc: Vec<Vec<f32>> = (0..128).map(|i| random_vec(dim, i + 100)).collect();

    let query_refs: Vec<&[f32]> = query.iter().map(|v| v.as_slice()).collect();
    let doc_refs: Vec<&[f32]> = doc.iter().map(|v| v.as_slice()).collect();

    g.bench_function("32x128x128", |bench| {
        bench.iter(|| black_box(simd::maxsim(&query_refs, &doc_refs)));
    });

    g.finish();
}

fn bench_matryoshka(c: &mut Criterion) {
    let mut g = c.benchmark_group("matryoshka");

    let dim = 768;
    let head = 64;
    let n_candidates = 100;

    let query = random_vec(dim, 0);
    let candidates: Vec<(&str, f32)> = (0..n_candidates)
        .map(|i| {
            (
                Box::leak(format!("d{i}").into_boxed_str()) as &str,
                0.9 - i as f32 * 0.01,
            )
        })
        .collect();
    let docs: Vec<(&str, Vec<f32>)> = candidates
        .iter()
        .map(|(id, _)| (*id, random_vec(dim, id.as_ptr() as u64)))
        .collect();

    g.bench_function("100x768_head64", |bench| {
        bench.iter(|| black_box(matryoshka::refine(&candidates, &query, &docs, head)));
    });

    g.finish();
}

fn bench_colbert(c: &mut Criterion) {
    let mut g = c.benchmark_group("colbert");

    let dim = 128;
    let n_docs = 100;

    let query: Vec<Vec<f32>> = (0..32).map(|i| random_vec(dim, i)).collect();
    let docs: Vec<(&str, Vec<Vec<f32>>)> = (0..n_docs)
        .map(|i| {
            let id = Box::leak(format!("d{i}").into_boxed_str()) as &str;
            let tokens: Vec<Vec<f32>> = (0..64)
                .map(|j| random_vec(dim, (i * 100 + j) as u64))
                .collect();
            (id, tokens)
        })
        .collect();

    g.bench_function("100docs_32qtok_64dtok", |bench| {
        bench.iter(|| black_box(colbert::rank(&query, &docs)));
    });

    g.finish();
}

fn bench_pool_tokens(c: &mut Criterion) {
    let mut g = c.benchmark_group("pool_tokens");

    let dim = 128;

    // Typical document: 64 tokens
    for &n_tokens in &[32, 64, 128] {
        let tokens: Vec<Vec<f32>> = (0..n_tokens).map(|i| random_vec(dim, i as u64)).collect();

        // Clustering-based pooling (factor 2)
        g.bench_with_input(
            BenchmarkId::new("clustering_f2", n_tokens),
            &tokens,
            |bench, toks| {
                bench.iter(|| black_box(colbert::pool_tokens(toks, 2)));
            },
        );

        // Clustering-based pooling (factor 3)
        g.bench_with_input(
            BenchmarkId::new("clustering_f3", n_tokens),
            &tokens,
            |bench, toks| {
                bench.iter(|| black_box(colbert::pool_tokens(toks, 3)));
            },
        );

        // Sequential pooling (factor 2)
        g.bench_with_input(
            BenchmarkId::new("sequential_f2", n_tokens),
            &tokens,
            |bench, toks| {
                bench.iter(|| black_box(colbert::pool_tokens_sequential(toks, 2)));
            },
        );

        // Sequential pooling (factor 4)
        g.bench_with_input(
            BenchmarkId::new("sequential_f4", n_tokens),
            &tokens,
            |bench, toks| {
                bench.iter(|| black_box(colbert::pool_tokens_sequential(toks, 4)));
            },
        );

        // Adaptive pooling (auto-selects method)
        g.bench_with_input(
            BenchmarkId::new("adaptive_f2", n_tokens),
            &tokens,
            |bench, toks| {
                bench.iter(|| black_box(colbert::pool_tokens_adaptive(toks, 2)));
            },
        );

        g.bench_with_input(
            BenchmarkId::new("adaptive_f4", n_tokens),
            &tokens,
            |bench, toks| {
                bench.iter(|| black_box(colbert::pool_tokens_adaptive(toks, 4)));
            },
        );
    }

    g.finish();
}

fn bench_maxsim_pooled(c: &mut Criterion) {
    let mut g = c.benchmark_group("maxsim_pooled");

    let dim = 128;
    let query: Vec<Vec<f32>> = (0..32).map(|i| random_vec(dim, i)).collect();
    let doc: Vec<Vec<f32>> = (0..128).map(|i| random_vec(dim, i + 1000)).collect();

    let query_refs: Vec<&[f32]> = query.iter().map(Vec::as_slice).collect();

    // Baseline: no pooling
    let doc_refs: Vec<&[f32]> = doc.iter().map(Vec::as_slice).collect();
    g.bench_function("no_pool_128tok", |bench| {
        bench.iter(|| black_box(simd::maxsim(&query_refs, &doc_refs)));
    });

    // Pooled: factor 2 (64 tokens)
    let pooled_2 = colbert::pool_tokens(&doc, 2);
    let p2_refs: Vec<&[f32]> = pooled_2.iter().map(Vec::as_slice).collect();
    g.bench_function("pool_f2_64tok", |bench| {
        bench.iter(|| black_box(simd::maxsim(&query_refs, &p2_refs)));
    });

    // Pooled: factor 4 (32 tokens)
    let pooled_4 = colbert::pool_tokens_sequential(&doc, 4);
    let p4_refs: Vec<&[f32]> = pooled_4.iter().map(Vec::as_slice).collect();
    g.bench_function("pool_f4_32tok", |bench| {
        bench.iter(|| black_box(simd::maxsim(&query_refs, &p4_refs)));
    });

    g.finish();
}

fn bench_mmr(c: &mut Criterion) {
    let mut g = c.benchmark_group("mmr");

    for &n in &[50, 100, 200] {
        // Generate candidates with scores
        let candidates: Vec<(usize, f32)> =
            (0..n).map(|i| (i, 1.0 - i as f32 / n as f32)).collect();

        // Generate similarity matrix (flattened, row-major)
        let sim: Vec<f32> = (0..n)
            .flat_map(|i| {
                (0..n).map(move |j| {
                    if i == j {
                        1.0
                    } else {
                        0.5 - (i as f32 - j as f32).abs() / (2.0 * n as f32)
                    }
                })
            })
            .collect();

        let config = diversity::MmrConfig::default().with_k(10);

        g.bench_with_input(BenchmarkId::new("k10", n), &n, |bench, _| {
            bench.iter(|| black_box(diversity::mmr(&candidates, &sim, config)));
        });
    }

    g.finish();
}

fn bench_dpp(c: &mut Criterion) {
    let mut g = c.benchmark_group("dpp");

    // DPP uses embeddings directly (not a similarity matrix)
    let dim = 128;

    for &n in &[50, 100, 200] {
        let candidates: Vec<(usize, f32)> =
            (0..n).map(|i| (i, 1.0 - i as f32 / n as f32)).collect();

        let embeddings: Vec<Vec<f32>> = (0..n).map(|i| random_vec(dim, i as u64)).collect();

        let config = diversity::DppConfig::default().with_k(10);

        g.bench_with_input(BenchmarkId::new("k10_d128", n), &n, |bench, _| {
            bench.iter(|| black_box(diversity::dpp(&candidates, &embeddings, config)));
        });
    }

    g.finish();
}

criterion_group!(
    benches,
    bench_simd,
    bench_maxsim,
    bench_matryoshka,
    bench_colbert,
    bench_pool_tokens,
    bench_maxsim_pooled,
    bench_mmr,
    bench_dpp
);
criterion_main!(benches);
