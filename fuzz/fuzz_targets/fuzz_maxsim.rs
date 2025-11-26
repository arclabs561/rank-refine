#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use rank_refine::simd;

#[derive(Arbitrary, Debug)]
struct MaxSimInput {
    query: Vec<Vec<f32>>,
    doc: Vec<Vec<f32>>,
}

fuzz_target!(|input: MaxSimInput| {
    let query_refs: Vec<&[f32]> = input.query.iter().map(Vec::as_slice).collect();
    let doc_refs: Vec<&[f32]> = input.doc.iter().map(Vec::as_slice).collect();
    
    // Should not panic on any input
    let _ = simd::maxsim(&query_refs, &doc_refs);
    let _ = simd::maxsim_cosine(&query_refs, &doc_refs);
});
