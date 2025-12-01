# Integration Guides

This document provides integration examples for common RAG/search stacks.

## Python Integration

### Using with LangChain

```python
from langchain.embeddings import OpenAIEmbeddings
import rank_refine
import numpy as np

# Get embeddings
embeddings = OpenAIEmbeddings()
query_embedding = embeddings.embed_query(query)
doc_embeddings = embeddings.embed_documents(documents)

# Dense reranking
scores = [
    rank_refine.cosine(query_embedding, doc_emb)
    for doc_emb in doc_embeddings
]

# MaxSim reranking (if you have token-level embeddings)
query_tokens = get_token_embeddings(query)  # Your tokenization
doc_tokens_list = [get_token_embeddings(doc) for doc in documents]

scores = rank_refine.maxsim_batch(query_tokens, doc_tokens_list)
```

### Using with LlamaIndex

```python
from llama_index import VectorStoreIndex
import rank_refine
import numpy as np

# Get initial retrieval results
results = index.retrieve(query, top_k=100)

# Extract embeddings
query_emb = get_query_embedding(query)
doc_embs = [get_doc_embedding(r.node) for r in results]

# Rerank with cosine similarity
scores = [
    rank_refine.cosine(query_emb, doc_emb)
    for doc_emb in doc_embs
]

# Reorder results
reranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
```

### Using with fastembed-rs (via Python bindings)

```python
# Assuming fastembed-rs has Python bindings
import fastembed
import rank_refine

# Encode query and documents
model = fastembed.TextEmbedding()
query_embedding = model.embed(query)
doc_embeddings = model.embed_batch(documents)

# Rerank
scores = [
    rank_refine.cosine(query_embedding, doc_emb)
    for doc_emb in doc_embeddings
]
```

## Rust Integration

### Using with qdrant

```rust
use qdrant_client::QdrantClient;
use rank_refine::simd::cosine;

// Get initial results from qdrant
let results = client
    .search_points(&SearchPoints {
        collection_name: "documents".to_string(),
        vector: query_embedding,
        limit: 100,
        ..Default::default()
    })
    .await?;

// Rerank with cosine similarity
let mut reranked: Vec<_> = results
    .result
    .iter()
    .map(|p| {
        let score = cosine(&query_embedding, &p.vector);
        (p.id.clone(), score)
    })
    .collect();

reranked.sort_by(|a, b| b.1.total_cmp(&a.1));
```

### Using with fastembed-rs

```rust
use fastembed::{TextEmbedding, EmbeddingModel};
use rank_refine::simd::{cosine, maxsim_vecs};

// Encode query
let model = TextEmbedding::try_new(EmbeddingModel::AllMiniLML6V2)?;
let query_embedding = model.embed(&[query], None)?;

// Encode documents
let doc_embeddings = model.embed(&documents, None)?;

// Dense reranking
let scores: Vec<f32> = doc_embeddings
    .iter()
    .map(|doc_emb| cosine(&query_embedding[0], doc_emb))
    .collect();

// MaxSim reranking (if model supports token-level embeddings)
let query_tokens = model.embed_tokens(&[query])?;
let doc_tokens_list: Vec<Vec<Vec<f32>>> = documents
    .iter()
    .map(|doc| model.embed_tokens(&[doc.clone()]).unwrap())
    .collect();

let maxsim_scores: Vec<f32> = doc_tokens_list
    .iter()
    .map(|doc_tokens| maxsim_vecs(&query_tokens[0], doc_tokens))
    .collect();
```

### Using with weaviate

```rust
use weaviate_community::WeaviateClient;
use rank_refine::simd::cosine;

// Get initial results
let results = weaviate_client
    .query()
    .get()
    .with_near_vector(query_embedding)
    .with_limit(100)
    .build()
    .execute()
    .await?;

// Rerank with cosine similarity
let mut reranked: Vec<_> = results
    .iter()
    .map(|r| {
        let score = cosine(&query_embedding, &r.vector);
        (r.id.clone(), score)
    })
    .collect();

reranked.sort_by(|a, b| b.1.total_cmp(&a.1));
```

## JavaScript/TypeScript Integration

### Using with WebAssembly

See `examples/webassembly.rs` for build instructions and JavaScript API design.

### Using via REST API

Create a Rust service:

```rust
use actix_web::{web, App, HttpServer, Result};
use rank_refine::simd::cosine;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct RerankRequest {
    query: Vec<f32>,
    documents: Vec<Vec<f32>>,
}

#[derive(Serialize)]
struct RerankResponse {
    scores: Vec<f32>,
}

async fn rerank(req: web::Json<RerankRequest>) -> Result<web::Json<RerankResponse>> {
    let scores: Vec<f32> = req.documents
        .iter()
        .map(|doc| cosine(&req.query, doc))
        .collect();
    Ok(web::Json(RerankResponse { scores }))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .route("/rerank", web::post().to(rerank))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
```

Then call from JavaScript:

```javascript
const response = await fetch('http://localhost:8080/rerank', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        query: [1.0, 0.0, 0.5],
        documents: [
            [0.9, 0.1, 0.4],
            [0.7, 0.3, 0.6]
        ]
    })
});
const { scores } = await response.json();
```

