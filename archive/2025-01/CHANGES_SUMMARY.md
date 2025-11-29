# Changes Summary: Humble, Simple, Plain, Objective Wording

## Changes Made

### rank-refine

1. **Subtitle**: Added key terms (MaxSim, ColBERT, MMR, DPP) for SEO
2. **Removed bold emphasis**: Changed "**The Problem**" → "Problem", "**Solution**" → "Solution"
3. **Simplified language**: 
   - "Traditional dense retrieval" → "Dense retrieval"
   - "This works well for broad topic matching" → "This works for broad matching"
   - "enabling precise reranking" → "Useful for reranking in RAG pipelines"
4. **Removed checkmarks**: Changed "✅" and "❌" to plain lists
5. **Added definitions**: "MaxSim scores token-level alignment" before formula
6. **Benchmark context**: Added "These timings enable real-time reranking of 100-1000 candidates"
7. **Objective tone**: Removed marketing language, kept facts

### rank-fusion

1. **Subtitle**: Added key terms (RRF, CombMNZ, Borda, DBSF) for SEO
2. **Removed bold emphasis**: Changed "**The Problem**" → "Problem", "**RRF Solution**" → "RRF (Reciprocal Rank Fusion)"
3. **Simplified language**:
   - "But how do you merge their results?" → "This requires merging their results"
   - "Simple normalization" → "Normalization"
   - "making hybrid search robust and zero-configuration" → "No normalization needed, works with any score distribution"
4. **Added definitions**: "RRF (Reciprocal Rank Fusion): Ignores score magnitudes..." before formula
5. **Benchmark context**: Added "These timings are suitable for real-time fusion of 100-1000 item lists"
6. **Objective tone**: Removed marketing language, kept facts
7. **Removed bold from examples**: "**0.0331**" → "0.0331", "← wins!" → "(wins)"

## Style Principles Applied

1. **Humble**: No claims of "best" or "fastest", just facts
2. **Simple**: Plain language, no jargon without explanation
3. **Plain**: Removed bold emphasis, checkmarks, marketing language
4. **Objective**: Facts over claims, measurements over opinions

## Remaining Work

GitHub repository descriptions should be updated to match Cargo.toml descriptions (most accurate):
- rank-refine: "SIMD-accelerated MaxSim, cosine, diversity (MMR/DPP) for vector search and RAG pipelines"
- rank-fusion: "Rank fusion algorithms for hybrid search — RRF, ISR, CombMNZ, Borda, DBSF. Zero dependencies."

