#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use rank_refine::simd;

#[derive(Arbitrary, Debug)]
struct VecPair {
    a: Vec<f32>,
    b: Vec<f32>,
}

fuzz_target!(|input: VecPair| {
    // Should not panic on any input
    let _ = simd::dot(&input.a, &input.b);
    let _ = simd::cosine(&input.a, &input.b);
});
