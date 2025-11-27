# Outreach: Integration Opportunities

## Target Projects

### swiftide (RAG pipeline framework)

**Repository**: https://github.com/bosun-ai/swiftide  
**Stars**: ~2.5k  
**Integration**: Diversity selection for query results

**Opportunity**: They use fastembed for embeddings but have no diversity selection. MMR/DPP would help reduce redundant chunks in RAG context.

**Draft Issue**: See `/rank-fusion/OUTREACH.md`

### pylate-rs (ColBERT inference)

**Repository**: https://github.com/lightonai/pylate-rs  
**Stars**: 67  
**Integration**: Token pooling for storage reduction

**Opportunity**: They do ColBERT inference with Candle. Our token pooling could reduce storage for their embeddings.

### qdrant (Vector database)

**Repository**: https://github.com/qdrant/qdrant  
**Stars**: 27k  
**Status**: ✅ Already has ColBERT/MaxSim (completed July 2024)

They implemented their own MaxSim. No integration opportunity, but validates the use case.

## Strategy

### Path A: Reference Library
- Focus on documentation quality
- Target citations/forks over downloads
- Let projects discover organically

### Path B: Infrastructure Primitive
- PR token pooling to pylate-rs
- PR diversity selection to swiftide
- Offer SIMD code for vendoring

## Vendoring Guide

For projects that want to vendor rather than depend:

**rank-refine SIMD code** (`src/simd.rs`):
- ~600 lines, self-contained
- AVX2+FMA / NEON with portable fallback
- No dependencies

**rank-fusion algorithms** (`src/lib.rs`):
- ~2000 lines, self-contained
- Zero dependencies
- All algorithms in one file

Both are MIT/Apache-2.0 licensed.

