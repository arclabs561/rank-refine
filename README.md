# rank-refine

SIMD-accelerated similarity scoring for vector search and RAG. Provides MaxSim (ColBERT/ColPali), cosine similarity, diversity selection (MMR, DPP), token pooling, token-level alignment/highlighting, and Matryoshka refinement. Supports both text (ColBERT) and multimodal (ColPali) late interaction.

This repository contains a Cargo workspace with multiple crates:

- **[`rank-refine`](rank-refine/)** — Core library (SIMD-accelerated)
- **[`rank-refine-python`](rank-refine-python/)** — Python bindings using PyO3
- **[`fuzz`](fuzz/)** — Fuzzing targets

[![CI](https://github.com/arclabs561/rank-refine/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rank-refine/actions)
[![Crates.io](https://img.shields.io/crates/v/rank-refine.svg)](https://crates.io/crates/rank-refine)
[![Docs](https://docs.rs/rank-refine/badge.svg)](https://docs.rs/rank-refine)

## Quick Start

### Rust

```bash
cargo add rank-refine
```

```rust
use rank_refine::simd;

// Dense scoring
let score = simd::cosine(&[1.0, 0.0], &[0.707, 0.707]);

// Late interaction (ColBERT MaxSim)
let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
let doc = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
let score = simd::maxsim_vecs(&query, &doc);
```

### Python

**Using uv (recommended):**

```bash
cd rank-refine-python
uv venv
source .venv/bin/activate
uv tool install maturin
maturin develop --uv
```

**Or using pip:**

```bash
cd rank-refine-python
pip install maturin
maturin develop --release
```

```python
import rank_refine

# Dense cosine similarity
query = [1.0, 0.0]
doc = [0.707, 0.707]
score = rank_refine.cosine(query, doc)

# MaxSim (late interaction)
query_tokens = [[1.0, 0.0], [0.0, 1.0]]
doc_tokens = [[0.9, 0.1], [0.1, 0.9]]
score = rank_refine.maxsim_vecs(query_tokens, doc_tokens)
```

### Node.js / WebAssembly

**Install from npm:**

```bash
npm install @arclabs561/rank-refine
```

**Usage in Node.js:**

```javascript
const { cosine, maxsim_vecs } = require('@arclabs561/rank-refine');

// Dense cosine similarity
const query = new Float32Array([1.0, 0.0]);
const doc = new Float32Array([0.707, 0.707]);
const score = cosine(query, doc);

// MaxSim (late interaction / ColBERT)
const query_tokens = [
  new Float32Array([1.0, 0.0]),
  new Float32Array([0.0, 1.0])
];
const doc_tokens = [
  new Float32Array([0.9, 0.1]),
  new Float32Array([0.1, 0.9])
];
const maxsim_score = maxsim_vecs(query_tokens, doc_tokens);
```

**Usage in TypeScript:**

```typescript
import { cosine, maxsim_vecs } from '@arclabs561/rank-refine';

const query = new Float32Array([1.0, 0.0]);
const doc = new Float32Array([0.707, 0.707]);
const score = cosine(query, doc);
```

**Usage in Browser (ES Modules):**

```javascript
import init, { cosine, maxsim_vecs } from '@arclabs561/rank-refine';

async function computeSimilarity() {
  // Initialize WASM module
  await init();
  
  // Dense similarity
  const query = new Float32Array([1.0, 0.0]);
  const doc = new Float32Array([0.707, 0.707]);
  const score = cosine(query, doc);
  
  // MaxSim for late interaction
  const query_tokens = [
    new Float32Array([1.0, 0.0]),
    new Float32Array([0.0, 1.0])
  ];
  const doc_tokens = [
    new Float32Array([0.9, 0.1]),
    new Float32Array([0.1, 0.9])
  ];
  const maxsim_score = maxsim_vecs(query_tokens, doc_tokens);
  
  console.log("Cosine:", score);
  console.log("MaxSim:", maxsim_score);
}

computeSimilarity();
```

## Documentation

- [Core crate documentation](rank-refine/README.md)
- [Python bindings](rank-refine-python/README.md)
- [Integration guide](rank-refine/INTEGRATION.md)
- [Design principles](rank-refine/DESIGN.md)

## Development

```bash
# Build core crate (fast, no Python required)
cargo build -p rank-refine

# Build all workspace members
cargo build --workspace

# Test core crate
cargo test -p rank-refine

# Test all workspace members
cargo test --workspace

# Check Python bindings (requires Python installed)
cargo check -p rank-refine-python

# Build Python bindings with uv
cd rank-refine-python
uv venv
source .venv/bin/activate
uv tool install maturin
maturin develop --uv
```

## Workspace Structure

This repository uses a Cargo workspace to organize the codebase:

- **Shared target directory** — All crates compile to one `target/` directory
- **Workspace inheritance** — Dependencies and versions defined once at workspace root
- **Path dependencies** — Python crate depends on core via path (no version conflicts)
- **Default members** — Only core crate builds by default (`cargo build`)

