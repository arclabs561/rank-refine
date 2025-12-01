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

use ::rank_refine::simd;
use pyo3::prelude::*;
use pyo3::types::PyList;

/// Python module for rank-refine.
#[pymodule]
#[pyo3(name = "rank_refine")]
fn rank_refine_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core SIMD functions (with user-friendly names)
    let cosine_func = wrap_pyfunction!(cosine_py, m)?;
    let dot_func = wrap_pyfunction!(dot_py, m)?;
    let norm_func = wrap_pyfunction!(norm_py, m)?;
    m.add("cosine", cosine_func)?;
    m.add("dot", dot_func)?;
    m.add("norm", norm_func)?;

    // MaxSim functions (with user-friendly names)
    let maxsim_vecs_func = wrap_pyfunction!(maxsim_vecs_py, m)?;
    let maxsim_cosine_vecs_func = wrap_pyfunction!(maxsim_cosine_vecs_py, m)?;
    let maxsim_batch_func = wrap_pyfunction!(maxsim_batch_py, m)?;
    m.add("maxsim_vecs", maxsim_vecs_func)?;
    m.add("maxsim_cosine_vecs", maxsim_cosine_vecs_func)?;
    m.add("maxsim_batch", maxsim_batch_func)?;

    Ok(())
}

/// Cosine similarity between two vectors.
///
/// # Arguments
/// * `a`: First vector (list of f32)
/// * `b`: Second vector (list of f32)
///
/// # Returns
/// Cosine similarity score (f32, -1.0 to 1.0)
#[pyfunction]
fn cosine_py(_py: Python<'_>, a: &Bound<'_, PyList>, b: &Bound<'_, PyList>) -> PyResult<f32> {
    let a_vec: Vec<f32> = a.extract()?;
    let b_vec: Vec<f32> = b.extract()?;
    Ok(simd::cosine(&a_vec, &b_vec))
}

/// Dot product of two vectors.
///
/// # Arguments
/// * `a`: First vector (list of f32)
/// * `b`: Second vector (list of f32)
///
/// # Returns
/// Dot product (f32)
#[pyfunction]
fn dot_py(_py: Python<'_>, a: &Bound<'_, PyList>, b: &Bound<'_, PyList>) -> PyResult<f32> {
    let a_vec: Vec<f32> = a.extract()?;
    let b_vec: Vec<f32> = b.extract()?;
    Ok(simd::dot(&a_vec, &b_vec))
}

/// L2 norm of a vector.
///
/// # Arguments
/// * `v`: Vector (list of f32)
///
/// # Returns
/// L2 norm (f32)
#[pyfunction]
fn norm_py(_py: Python<'_>, v: &Bound<'_, PyList>) -> PyResult<f32> {
    let v_vec: Vec<f32> = v.extract()?;
    Ok(simd::norm(&v_vec))
}

/// MaxSim (late interaction) scoring for token-level embeddings.
///
/// # Arguments
/// * `query_tokens`: Query token embeddings (list of lists, shape: [n_tokens, dim])
/// * `doc_tokens`: Document token embeddings (list of lists, shape: [m_tokens, dim])
///
/// # Returns
/// MaxSim score (f32)
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
///
/// # Arguments
/// * `query_tokens`: Query token embeddings (list of lists)
/// * `doc_tokens`: Document token embeddings (list of lists)
///
/// # Returns
/// MaxSim cosine score (f32)
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
///
/// # Arguments
/// * `query_tokens`: Query token embeddings (list of lists)
/// * `doc_tokens_list`: List of document token embeddings (list of lists of lists)
///
/// # Returns
/// List of MaxSim scores (one per document)
#[pyfunction]
fn maxsim_batch_py(
    py: Python<'_>,
    query_tokens: &Bound<'_, PyList>,
    doc_tokens_list: &Bound<'_, PyList>,
) -> PyResult<Py<PyList>> {
    let query_vecs: Vec<Vec<f32>> = py_list_of_lists_to_vec(query_tokens)?;

    // Convert documents to Vec<Vec<Vec<f32>>>
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

    // Call Rust function
    let scores = simd::maxsim_batch(&query_vecs, &doc_vecs_list);

    // Convert back to Python list
    let result = PyList::empty_bound(py);
    for score in scores {
        result.append(score)?;
    }
    Ok(result.into())
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
