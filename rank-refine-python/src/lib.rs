//! Python bindings for rank-refine using PyO3.
//!
//! Provides a Python API that mirrors the Rust API, enabling seamless
//! integration with Python RAG/search stacks.
//!
//! # Usage
//!
//! ```python
//! import rank_refine
//!
//! # Dense cosine similarity
//! query = [1.0, 0.0]
//! doc = [0.707, 0.707]
//! score = rank_refine.cosine(query, doc)
//!
//! # MaxSim (late interaction)
//! query_tokens = [[1.0, 0.0], [0.0, 1.0]]
//! doc_tokens = [[0.9, 0.1], [0.1, 0.9]]
//! score = rank_refine.maxsim_vecs(query_tokens, doc_tokens)
//! ```

// TODO: Remove allow(deprecated) when upgrading to pyo3 0.25+ which uses IntoPyObject
#![allow(deprecated)]

use ::rank_refine::simd;
use ::rank_refine::{colbert, diversity, matryoshka, RefineConfig};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

/// Python module for rank-refine.
#[pymodule]
#[pyo3(name = "rank_refine")]
fn rank_refine_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core SIMD functions
    m.add_function(wrap_pyfunction!(cosine_py, m)?)?;
    m.add_function(wrap_pyfunction!(dot_py, m)?)?;
    m.add_function(wrap_pyfunction!(norm_py, m)?)?;

    // MaxSim functions
    m.add_function(wrap_pyfunction!(maxsim_vecs_py, m)?)?;
    m.add_function(wrap_pyfunction!(maxsim_cosine_vecs_py, m)?)?;
    m.add_function(wrap_pyfunction!(maxsim_batch_py, m)?)?;

    // Alignment functions
    m.add_function(wrap_pyfunction!(maxsim_alignments_vecs_py, m)?)?;
    m.add_function(wrap_pyfunction!(maxsim_alignments_cosine_vecs_py, m)?)?;
    m.add_function(wrap_pyfunction!(highlight_matches_vecs_py, m)?)?;

    // Diversity functions
    m.add_function(wrap_pyfunction!(mmr_py, m)?)?;
    m.add_function(wrap_pyfunction!(mmr_cosine_py, m)?)?;
    m.add_function(wrap_pyfunction!(dpp_py, m)?)?;

    // ColBERT functions
    m.add_function(wrap_pyfunction!(pool_tokens_py, m)?)?;
    m.add_function(wrap_pyfunction!(colbert_alignments_py, m)?)?;
    m.add_function(wrap_pyfunction!(colbert_highlight_py, m)?)?;

    // Matryoshka functions
    m.add_function(wrap_pyfunction!(matryoshka_try_refine_py, m)?)?;

    // Configuration classes
    m.add_class::<MmrConfigPy>()?;
    m.add_class::<DppConfigPy>()?;

    Ok(())
}

/// Cosine similarity between two vectors.
#[pyfunction]
fn cosine_py(_py: Python<'_>, a: &Bound<'_, PyList>, b: &Bound<'_, PyList>) -> PyResult<f32> {
    let a_vec: Vec<f32> = a.extract()?;
    let b_vec: Vec<f32> = b.extract()?;
    Ok(simd::cosine(&a_vec, &b_vec))
}

/// Dot product of two vectors.
#[pyfunction]
fn dot_py(_py: Python<'_>, a: &Bound<'_, PyList>, b: &Bound<'_, PyList>) -> PyResult<f32> {
    let a_vec: Vec<f32> = a.extract()?;
    let b_vec: Vec<f32> = b.extract()?;
    Ok(simd::dot(&a_vec, &b_vec))
}

/// L2 norm of a vector.
#[pyfunction]
fn norm_py(_py: Python<'_>, v: &Bound<'_, PyList>) -> PyResult<f32> {
    let v_vec: Vec<f32> = v.extract()?;
    Ok(simd::norm(&v_vec))
}

/// MaxSim (late interaction) scoring for token-level embeddings.
#[pyfunction]
fn maxsim_vecs_py(
    _py: Python<'_>,
    query_tokens: &Bound<'_, PyList>,
    doc_tokens: &Bound<'_, PyList>,
) -> PyResult<f32> {
    let query_vecs: Vec<Vec<f32>> = py_list_of_lists_to_vec(query_tokens)?;
    let doc_vecs: Vec<Vec<f32>> = py_list_of_lists_to_vec(doc_tokens)?;

    if !query_vecs.is_empty() && !doc_vecs.is_empty() {
        let query_dim = query_vecs[0].len();
        let doc_dim = doc_vecs[0].len();
        if query_dim != doc_dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Dimension mismatch: query tokens have {} dims, doc tokens have {} dims",
                query_dim, doc_dim
            )));
        }
    }

    Ok(simd::maxsim_vecs(&query_vecs, &doc_vecs))
}

/// MaxSim with cosine similarity variant.
#[pyfunction]
fn maxsim_cosine_vecs_py(
    _py: Python<'_>,
    query_tokens: &Bound<'_, PyList>,
    doc_tokens: &Bound<'_, PyList>,
) -> PyResult<f32> {
    let query_vecs: Vec<Vec<f32>> = py_list_of_lists_to_vec(query_tokens)?;
    let doc_vecs: Vec<Vec<f32>> = py_list_of_lists_to_vec(doc_tokens)?;

    if !query_vecs.is_empty() && !doc_vecs.is_empty() {
        let query_dim = query_vecs[0].len();
        let doc_dim = doc_vecs[0].len();
        if query_dim != doc_dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Dimension mismatch: query tokens have {} dims, doc tokens have {} dims",
                query_dim, doc_dim
            )));
        }
    }

    Ok(simd::maxsim_cosine_vecs(&query_vecs, &doc_vecs))
}

/// Batch MaxSim scoring for multiple documents.
#[pyfunction]
fn maxsim_batch_py(
    py: Python<'_>,
    query_tokens: &Bound<'_, PyList>,
    doc_tokens_list: &Bound<'_, PyList>,
) -> PyResult<Py<PyList>> {
    let query_vecs: Vec<Vec<f32>> = py_list_of_lists_to_vec(query_tokens)?;

    let mut doc_vecs_list = Vec::new();
    for item in doc_tokens_list.iter() {
        let doc_list = item.downcast::<PyList>()?;
        let doc_vecs: Vec<Vec<f32>> = py_list_of_lists_to_vec(doc_list)?;

        if !query_vecs.is_empty() && !doc_vecs.is_empty() {
            let query_dim = query_vecs[0].len();
            let doc_dim = doc_vecs[0].len();
            if query_dim != doc_dim {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Dimension mismatch: query has {} dims, doc has {} dims",
                    query_dim, doc_dim
                )));
            }
        }

        doc_vecs_list.push(doc_vecs);
    }

    let scores = simd::maxsim_batch(&query_vecs, &doc_vecs_list);

    let result = PyList::empty(py);
    for score in scores {
        result.append(score)?;
    }
    Ok(result.into())
}

/// Token-level alignments for MaxSim.
///
/// Returns list of (query_idx, doc_idx, similarity_score) tuples.
#[pyfunction]
fn maxsim_alignments_vecs_py(
    py: Python<'_>,
    query_tokens: &Bound<'_, PyList>,
    doc_tokens: &Bound<'_, PyList>,
) -> PyResult<Py<PyList>> {
    let query_vecs: Vec<Vec<f32>> = py_list_of_lists_to_vec(query_tokens)?;
    let doc_vecs: Vec<Vec<f32>> = py_list_of_lists_to_vec(doc_tokens)?;

    if !query_vecs.is_empty() && !doc_vecs.is_empty() {
        let query_dim = query_vecs[0].len();
        let doc_dim = doc_vecs[0].len();
        if query_dim != doc_dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Dimension mismatch: query tokens have {} dims, doc tokens have {} dims",
                query_dim, doc_dim
            )));
        }
    }

    let alignments = simd::maxsim_alignments_vecs(&query_vecs, &doc_vecs);

    let result = PyList::empty(py);
    for (q_idx, d_idx, score) in alignments {
        let tuple = PyTuple::new(py, &[q_idx.into_py(py), d_idx.into_py(py), score.into_py(py)])?;
        result.append(tuple)?;
    }
    Ok(result.into())
}

/// Token-level alignments with cosine similarity.
#[pyfunction]
fn maxsim_alignments_cosine_vecs_py(
    py: Python<'_>,
    query_tokens: &Bound<'_, PyList>,
    doc_tokens: &Bound<'_, PyList>,
) -> PyResult<Py<PyList>> {
    let query_vecs: Vec<Vec<f32>> = py_list_of_lists_to_vec(query_tokens)?;
    let doc_vecs: Vec<Vec<f32>> = py_list_of_lists_to_vec(doc_tokens)?;

    if !query_vecs.is_empty() && !doc_vecs.is_empty() {
        let query_dim = query_vecs[0].len();
        let doc_dim = doc_vecs[0].len();
        if query_dim != doc_dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Dimension mismatch: query tokens have {} dims, doc tokens have {} dims",
                query_dim, doc_dim
            )));
        }
    }

    let alignments = simd::maxsim_alignments_cosine_vecs(&query_vecs, &doc_vecs);

    let result = PyList::empty(py);
    for (q_idx, d_idx, score) in alignments {
        let tuple = PyTuple::new(py, &[q_idx.into_py(py), d_idx.into_py(py), score.into_py(py)])?;
        result.append(tuple)?;
    }
    Ok(result.into())
}

/// Highlighted document token indices above threshold.
#[pyfunction]
fn highlight_matches_vecs_py(
    py: Python<'_>,
    query_tokens: &Bound<'_, PyList>,
    doc_tokens: &Bound<'_, PyList>,
    threshold: f32,
) -> PyResult<Py<PyList>> {
    let query_vecs: Vec<Vec<f32>> = py_list_of_lists_to_vec(query_tokens)?;
    let doc_vecs: Vec<Vec<f32>> = py_list_of_lists_to_vec(doc_tokens)?;

    if !query_vecs.is_empty() && !doc_vecs.is_empty() {
        let query_dim = query_vecs[0].len();
        let doc_dim = doc_vecs[0].len();
        if query_dim != doc_dim {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Dimension mismatch: query tokens have {} dims, doc tokens have {} dims",
                query_dim, doc_dim
            )));
        }
    }

    let highlighted = simd::highlight_matches_vecs(&query_vecs, &doc_vecs, threshold);

    let result = PyList::empty(py);
    for idx in highlighted {
        result.append(idx)?;
    }
    Ok(result.into())
}

/// Maximal Marginal Relevance (MMR) with precomputed similarity matrix.
#[pyfunction]
#[pyo3(signature = (candidates, similarity, lambda = 0.5, k = 10))]
fn mmr_py(
    py: Python<'_>,
    candidates: &Bound<'_, PyList>,
    similarity: &Bound<'_, PyList>,
    lambda: f32,
    k: usize,
) -> PyResult<Py<PyList>> {
    let candidates_vec: Vec<(String, f32)> = py_list_to_ranked(candidates)?;
    let similarity_vec: Vec<f32> = similarity.extract()?;

    let config = diversity::MmrConfig::new(lambda, k);
    let selected = diversity::try_mmr(&candidates_vec, &similarity_vec, config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

    let result = PyList::empty(py);
    for (id, score) in selected {
        let tuple = PyTuple::new(py, &[id.into_py(py), score.into_py(py)])?;
        result.append(tuple)?;
    }
    Ok(result.into())
}

/// MMR with cosine similarity computed from embeddings.
#[pyfunction]
#[pyo3(signature = (candidates, embeddings, *, lambda = 0.5, k = 10))]
fn mmr_cosine_py(
    py: Python<'_>,
    candidates: &Bound<'_, PyList>,
    embeddings: &Bound<'_, PyList>,
    lambda: f32,
    k: usize,
) -> PyResult<Py<PyList>> {
    let candidates_vec: Vec<(String, f32)> = py_list_to_ranked(candidates)?;
    let embeddings_vec: Vec<Vec<f32>> = py_list_of_lists_to_vec(embeddings)?;

    if candidates_vec.len() != embeddings_vec.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "candidates and embeddings must have same length",
        ));
    }

    let config = diversity::MmrConfig::new(lambda, k);
    let selected = diversity::mmr_cosine(&candidates_vec, &embeddings_vec, config);

    let result = PyList::empty(py);
    for (id, score) in selected {
        let tuple = PyTuple::new(py, &[id.into_py(py), score.into_py(py)])?;
        result.append(tuple)?;
    }
    Ok(result.into())
}

/// Determinantal Point Process (DPP) diversity selection.
#[pyfunction]
#[pyo3(signature = (candidates, embeddings, alpha = 1.0, k = 10))]
fn dpp_py(
    py: Python<'_>,
    candidates: &Bound<'_, PyList>,
    embeddings: &Bound<'_, PyList>,
    alpha: f32,
    k: usize,
) -> PyResult<Py<PyList>> {
    let candidates_vec: Vec<(String, f32)> = py_list_to_ranked(candidates)?;
    let embeddings_vec: Vec<Vec<f32>> = py_list_of_lists_to_vec(embeddings)?;

    if candidates_vec.len() != embeddings_vec.len() {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "candidates and embeddings must have same length",
        ));
    }

    let config = diversity::DppConfig::new(k, alpha);
    let selected = diversity::dpp(&candidates_vec, &embeddings_vec, config);

    let result = PyList::empty(py);
    for (id, score) in selected {
        let tuple = PyTuple::new(py, &[id.into_py(py), score.into_py(py)])?;
        result.append(tuple)?;
    }
    Ok(result.into())
}

/// Pool tokens to reduce storage (clustering-based).
#[pyfunction]
fn pool_tokens_py(
    py: Python<'_>,
    tokens: &Bound<'_, PyList>,
    pool_factor: usize,
) -> PyResult<Py<PyList>> {
    let tokens_vec: Vec<Vec<f32>> = py_list_of_lists_to_vec(tokens)?;

    let pooled = colbert::pool_tokens(&tokens_vec, pool_factor)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

    let result = PyList::empty(py);
    for token in pooled {
        let token_list = PyList::empty(py);
        for val in token {
            token_list.append(val)?;
        }
        result.append(token_list)?;
    }
    Ok(result.into())
}

/// ColBERT token alignments (convenience wrapper).
#[pyfunction]
fn colbert_alignments_py(
    py: Python<'_>,
    query: &Bound<'_, PyList>,
    doc: &Bound<'_, PyList>,
) -> PyResult<Py<PyList>> {
    let query_vecs: Vec<Vec<f32>> = py_list_of_lists_to_vec(query)?;
    let doc_vecs: Vec<Vec<f32>> = py_list_of_lists_to_vec(doc)?;

    let alignments = colbert::alignments(&query_vecs, &doc_vecs);

    let result = PyList::empty(py);
    for (q_idx, d_idx, score) in alignments {
        let tuple = PyTuple::new(py, &[q_idx.into_py(py), d_idx.into_py(py), score.into_py(py)])?;
        result.append(tuple)?;
    }
    Ok(result.into())
}

/// ColBERT highlight (convenience wrapper).
#[pyfunction]
fn colbert_highlight_py(
    py: Python<'_>,
    query: &Bound<'_, PyList>,
    doc: &Bound<'_, PyList>,
    threshold: f32,
) -> PyResult<Py<PyList>> {
    let query_vecs: Vec<Vec<f32>> = py_list_of_lists_to_vec(query)?;
    let doc_vecs: Vec<Vec<f32>> = py_list_of_lists_to_vec(doc)?;

    let highlighted = colbert::highlight(&query_vecs, &doc_vecs, threshold);

    let result = PyList::empty(py);
    for idx in highlighted {
        result.append(idx)?;
    }
    Ok(result.into())
}

/// Matryoshka refinement (fallible version to avoid panics).
#[pyfunction]
#[pyo3(signature = (candidates, query, docs, head_dims, alpha = 0.5))]
fn matryoshka_try_refine_py(
    py: Python<'_>,
    candidates: &Bound<'_, PyList>,
    query: &Bound<'_, PyList>,
    docs: &Bound<'_, PyList>,
    head_dims: usize,
    alpha: f32,
) -> PyResult<Py<PyList>> {
    let candidates_vec: Vec<(String, f32)> = py_list_to_ranked(candidates)?;
    let query_vec: Vec<f32> = query.extract()?;
    let docs_vec: Vec<(String, Vec<f32>)> = py_list_to_docs(docs)?;

    let config = RefineConfig::default().with_alpha(alpha);
    let refined = matryoshka::try_refine(&candidates_vec, &query_vec, &docs_vec, head_dims, config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;

    let result = PyList::empty(py);
    for (id, score) in refined {
        let tuple = PyTuple::new(py, &[id.into_py(py), score.into_py(py)])?;
        result.append(tuple)?;
    }
    Ok(result.into())
}

/// Python wrapper for MmrConfig.
#[pyclass]
struct MmrConfigPy {
    inner: diversity::MmrConfig,
}

#[pymethods]
impl MmrConfigPy {
    #[new]
    #[pyo3(signature = (*, lambda = 0.5, k = 10))]
    fn new(lambda: f32, k: usize) -> Self {
        Self {
            inner: diversity::MmrConfig::new(lambda, k),
        }
    }

    #[getter]
    fn lambda(&self) -> f32 {
        self.inner.lambda
    }

    #[getter]
    fn k(&self) -> usize {
        self.inner.k
    }

    fn with_lambda(&self, lambda: f32) -> Self {
        Self {
            inner: self.inner.with_lambda(lambda),
        }
    }

    fn with_k(&self, k: usize) -> Self {
        Self {
            inner: self.inner.with_k(k),
        }
    }
}

/// Python wrapper for DppConfig.
#[pyclass]
struct DppConfigPy {
    inner: diversity::DppConfig,
}

#[pymethods]
impl DppConfigPy {
    #[new]
    #[pyo3(signature = (alpha = 1.0, k = 10))]
    fn new(alpha: f32, k: usize) -> Self {
        Self {
            inner: diversity::DppConfig::new(k, alpha),
        }
    }

    #[getter]
    fn alpha(&self) -> f32 {
        self.inner.alpha
    }

    #[getter]
    fn k(&self) -> usize {
        self.inner.k
    }

    fn with_alpha(&self, alpha: f32) -> Self {
        Self {
            inner: self.inner.with_alpha(alpha),
        }
    }

    fn with_k(&self, k: usize) -> Self {
        Self {
            inner: self.inner.with_k(k),
        }
    }
}

/// Helper to convert Python list of lists to Vec<Vec<f32>>.
fn py_list_of_lists_to_vec(py_list: &Bound<'_, PyList>) -> PyResult<Vec<Vec<f32>>> {
    let mut result = Vec::new();
    for item in py_list.iter() {
        let inner_list = item.downcast::<PyList>()?;
        let vec: Vec<f32> = inner_list.extract()?;
        result.push(vec);
    }
    Ok(result)
}

/// Helper to convert Python list of (id, score) tuples to Vec<(String, f32)>.
fn py_list_to_ranked(py_list: &Bound<'_, PyList>) -> PyResult<Vec<(String, f32)>> {
    let mut result = Vec::new();
    for item in py_list.iter() {
        let tuple = item.downcast::<PyTuple>()?;
        if tuple.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Expected (id, score) tuple",
            ));
        }
        let id: String = tuple.get_item(0)?.extract()?;
        let score: f32 = tuple.get_item(1)?.extract()?;
        result.push((id, score));
    }
    Ok(result)
}

/// Helper to convert Python list of (id, embedding) tuples to Vec<(String, Vec<f32>)>.
fn py_list_to_docs(py_list: &Bound<'_, PyList>) -> PyResult<Vec<(String, Vec<f32>)>> {
    let mut result = Vec::new();
    for item in py_list.iter() {
        let tuple = item.downcast::<PyTuple>()?;
        if tuple.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Expected (id, embedding) tuple",
            ));
        }
        let id: String = tuple.get_item(0)?.extract()?;
        let embedding: Vec<f32> = {
            let embedding_item = tuple.get_item(1)?;
            let embedding_list = embedding_item.downcast::<PyList>()?;
            embedding_list.extract()?
        };
        result.push((id, embedding));
    }
    Ok(result)
}
