# rank-refine-python

Python bindings for [`rank-refine`](../rank-refine/README.md) — SIMD-accelerated similarity scoring for vector search and RAG.

## Installation

### From PyPI

```bash
pip install rank-refine
```

### From source

```bash
# Using uv (recommended)
cd rank-refine-python
uv venv
source .venv/bin/activate
uv tool install maturin
maturin develop --uv

# Or using pip
pip install maturin
maturin develop --release
```

## Usage

```python
import rank_refine

# Dense cosine similarity
query = [1.0, 0.0]
doc = [0.707, 0.707]
score = rank_refine.cosine(query, doc)

# Dot product
score = rank_refine.dot(query, doc)

# L2 norm
norm = rank_refine.norm(query)

# MaxSim (late interaction / ColBERT)
query_tokens = [[1.0, 0.0], [0.0, 1.0]]
doc_tokens = [[0.9, 0.1], [0.1, 0.9]]
score = rank_refine.maxsim_vecs(query_tokens, doc_tokens)

# MaxSim with cosine similarity
score = rank_refine.maxsim_cosine_vecs(query_tokens, doc_tokens)

# Batch MaxSim scoring
doc_tokens_list = [
    [[0.9, 0.1], [0.1, 0.9]],
    [[0.8, 0.2], [0.2, 0.8]],
]
scores = rank_refine.maxsim_batch(query_tokens, doc_tokens_list)
```

## API

- `cosine(a, b)` — Cosine similarity between two vectors
- `dot(a, b)` — Dot product of two vectors
- `norm(v)` — L2 norm of a vector
- `maxsim_vecs(query_tokens, doc_tokens)` — MaxSim scoring
- `maxsim_cosine_vecs(query_tokens, doc_tokens)` — MaxSim with cosine similarity
- `maxsim_batch(query_tokens, doc_tokens_list)` — Batch MaxSim scoring

## See Also

- [Core crate documentation](../rank-refine/README.md)
- [Integration guide](../rank-refine/INTEGRATION.md)
