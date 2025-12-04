"""Comprehensive tests for rank-refine Python bindings."""

import pytest
import rank_refine
from typing import List, Tuple


# Test data fixtures
@pytest.fixture
def query_vec() -> List[float]:
    """Query vector."""
    return [1.0, 0.0, 0.0]


@pytest.fixture
def doc_vec() -> List[float]:
    """Document vector."""
    return [0.707, 0.707, 0.0]


@pytest.fixture
def query_tokens() -> List[List[float]]:
    """Query token embeddings."""
    return [[1.0, 0.0], [0.0, 1.0]]


@pytest.fixture
def doc_tokens() -> List[List[float]]:
    """Document token embeddings."""
    return [[0.9, 0.1], [0.1, 0.9]]


@pytest.fixture
def candidates() -> List[Tuple[str, float]]:
    """Candidates with relevance scores."""
    return [("doc1", 0.95), ("doc2", 0.90), ("doc3", 0.85)]


@pytest.fixture
def embeddings() -> List[List[float]]:
    """Embeddings for candidates."""
    return [
        [1.0, 0.0],  # doc1
        [0.9, 0.1],  # doc2 (similar to doc1)
        [0.0, 1.0],  # doc3 (different)
    ]


@pytest.fixture
def similarity_matrix() -> List[float]:
    """Flattened similarity matrix (row-major)."""
    # 3x3 matrix: doc1 vs [doc1, doc2, doc3], doc2 vs [...], doc3 vs [...]
    return [
        1.0, 0.9, 0.2,  # doc1 vs [doc1, doc2, doc3]
        0.9, 1.0, 0.3,  # doc2 vs [doc1, doc2, doc3]
        0.2, 0.3, 1.0,  # doc3 vs [doc1, doc2, doc3]
    ]


# Core SIMD Tests
class TestSIMD:
    """Tests for core SIMD functions."""

    def test_cosine(self, query_vec: List[float], doc_vec: List[float]):
        """Test cosine similarity."""
        score = rank_refine.cosine(query_vec, doc_vec)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_dot(self, query_vec: List[float], doc_vec: List[float]):
        """Test dot product."""
        score = rank_refine.dot(query_vec, doc_vec)
        assert isinstance(score, float)

    def test_norm(self, query_vec: List[float]):
        """Test L2 norm."""
        norm = rank_refine.norm(query_vec)
        assert isinstance(norm, float)
        assert norm >= 0.0


# MaxSim Tests
class TestMaxSim:
    """Tests for MaxSim (late interaction) functions."""

    def test_maxsim_vecs(
        self, query_tokens: List[List[float]], doc_tokens: List[List[float]]
    ):
        """Test MaxSim scoring."""
        score = rank_refine.maxsim_vecs(query_tokens, doc_tokens)
        assert isinstance(score, float)

    def test_maxsim_cosine_vecs(
        self, query_tokens: List[List[float]], doc_tokens: List[List[float]]
    ):
        """Test MaxSim with cosine."""
        score = rank_refine.maxsim_cosine_vecs(query_tokens, doc_tokens)
        assert isinstance(score, float)

    def test_maxsim_batch(
        self, query_tokens: List[List[float]]
    ):
        """Test batch MaxSim scoring."""
        doc_tokens_list = [
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.8, 0.2], [0.2, 0.8]],
        ]
        scores = rank_refine.maxsim_batch(query_tokens, doc_tokens_list)
        assert isinstance(scores, list)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)


# Alignment Tests
class TestAlignment:
    """Tests for token alignment functions."""

    def test_maxsim_alignments_vecs(
        self, query_tokens: List[List[float]], doc_tokens: List[List[float]]
    ):
        """Test MaxSim alignments."""
        alignments = rank_refine.maxsim_alignments_vecs(query_tokens, doc_tokens)
        assert isinstance(alignments, list)
        if len(alignments) > 0:
            assert all(
                isinstance(a, tuple) and len(a) == 3 for a in alignments
            )
            assert all(
                isinstance(a[0], int) and isinstance(a[1], int) and isinstance(a[2], float)
                for a in alignments
            )

    def test_maxsim_alignments_cosine_vecs(
        self, query_tokens: List[List[float]], doc_tokens: List[List[float]]
    ):
        """Test MaxSim alignments with cosine."""
        alignments = rank_refine.maxsim_alignments_cosine_vecs(query_tokens, doc_tokens)
        assert isinstance(alignments, list)

    def test_highlight_matches_vecs(
        self, query_tokens: List[List[float]], doc_tokens: List[List[float]]
    ):
        """Test highlight matches."""
        highlighted = rank_refine.highlight_matches_vecs(query_tokens, doc_tokens, 0.7)
        assert isinstance(highlighted, list)
        assert all(isinstance(idx, int) for idx in highlighted)


# Diversity Tests
class TestDiversity:
    """Tests for diversity selection functions."""

    def test_mmr(
        self,
        candidates: List[Tuple[str, float]],
        similarity_matrix: List[float],
    ):
        """Test MMR with precomputed similarity."""
        selected = rank_refine.mmr(candidates, similarity_matrix, lambda=0.5, k=2)
        assert isinstance(selected, list)
        assert len(selected) <= 2
        assert all(isinstance(item, tuple) and len(item) == 2 for item in selected)

    def test_mmr_cosine(
        self,
        candidates: List[Tuple[str, float]],
        embeddings: List[List[float]],
    ):
        """Test MMR with cosine similarity."""
        selected = rank_refine.mmr_cosine(candidates, embeddings, lambda=0.5, k=2)
        assert isinstance(selected, list)
        assert len(selected) <= 2

    def test_dpp(
        self,
        candidates: List[Tuple[str, float]],
        embeddings: List[List[float]],
    ):
        """Test DPP diversity selection."""
        selected = rank_refine.dpp(candidates, embeddings, alpha=1.0, k=2)
        assert isinstance(selected, list)
        assert len(selected) <= 2

    def test_mmr_empty_candidates(self):
        """Test MMR with empty candidates."""
        selected = rank_refine.mmr([], [], lambda=0.5, k=10)
        assert isinstance(selected, list)
        assert len(selected) == 0

    def test_mmr_dimension_mismatch_error(
        self, candidates: List[Tuple[str, float]]
    ):
        """Test MMR with wrong similarity matrix size raises error."""
        wrong_similarity = [1.0, 0.9, 0.2]  # Too small (should be 9 for 3 candidates)
        with pytest.raises(ValueError):
            rank_refine.mmr(candidates, wrong_similarity, lambda=0.5, k=2)


# ColBERT Tests
class TestColBERT:
    """Tests for ColBERT functions."""

    def test_pool_tokens(self, doc_tokens: List[List[float]]):
        """Test token pooling."""
        pooled = rank_refine.pool_tokens(doc_tokens, pool_factor=2)
        assert isinstance(pooled, list)
        assert len(pooled) <= len(doc_tokens)

    def test_pool_tokens_factor_one(self, doc_tokens: List[List[float]]):
        """Test token pooling with factor 1 (no pooling)."""
        pooled = rank_refine.pool_tokens(doc_tokens, pool_factor=1)
        assert isinstance(pooled, list)
        assert len(pooled) == len(doc_tokens)

    def test_pool_tokens_zero_factor_error(self, doc_tokens: List[List[float]]):
        """Test token pooling with factor 0 raises error."""
        with pytest.raises(ValueError):
            rank_refine.pool_tokens(doc_tokens, pool_factor=0)

    def test_colbert_alignments(
        self, query_tokens: List[List[float]], doc_tokens: List[List[float]]
    ):
        """Test ColBERT alignments."""
        alignments = rank_refine.colbert_alignments(query_tokens, doc_tokens)
        assert isinstance(alignments, list)

    def test_colbert_highlight(
        self, query_tokens: List[List[float]], doc_tokens: List[List[float]]
    ):
        """Test ColBERT highlight."""
        highlighted = rank_refine.colbert_highlight(query_tokens, doc_tokens, 0.7)
        assert isinstance(highlighted, list)
        assert all(isinstance(idx, int) for idx in highlighted)


# Matryoshka Tests
class TestMatryoshka:
    """Tests for Matryoshka refinement."""

    def test_matryoshka_try_refine(self):
        """Test Matryoshka refinement."""
        candidates = [("doc1", 0.9), ("doc2", 0.8)]
        query = [0.1, 0.2, 0.3, 0.4]  # 4 dims
        docs = [
            ("doc1", [0.1, 0.2, 0.35, 0.45]),  # tail: [0.35, 0.45]
            ("doc2", [0.1, 0.2, 0.29, 0.39]),  # tail: [0.29, 0.39]
        ]
        head_dims = 2  # First 2 dims are head, last 2 are tail

        refined = rank_refine.matryoshka_try_refine(
            candidates, query, docs, head_dims, alpha=0.5
        )
        assert isinstance(refined, list)
        assert len(refined) <= len(candidates)

    def test_matryoshka_invalid_head_dims_error(self):
        """Test Matryoshka with invalid head_dims raises error."""
        candidates = [("doc1", 0.9)]
        query = [0.1, 0.2]  # 2 dims
        docs = [("doc1", [0.1, 0.2])]
        head_dims = 2  # head_dims >= query.len() should error

        with pytest.raises(ValueError, match="head_dims"):
            rank_refine.matryoshka_try_refine(
                candidates, query, docs, head_dims, alpha=0.5
            )


# Configuration Classes Tests
class TestConfigClasses:
    """Tests for configuration classes."""

    def test_mmr_config(self):
        """Test MmrConfigPy class."""
        config = rank_refine.MmrConfigPy(lambda=0.7, k=5)
        assert config.lambda == 0.7
        assert config.k == 5
        config_with_lambda = config.with_lambda(0.3)
        assert config_with_lambda.lambda == 0.3

    def test_dpp_config(self):
        """Test DppConfigPy class."""
        config = rank_refine.DppConfigPy(alpha=2.0, k=5)
        assert config.alpha == 2.0
        assert config.k == 5
        config_with_alpha = config.with_alpha(1.5)
        assert config_with_alpha.alpha == 1.5


# Edge Cases and Error Handling
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_dimension_mismatch_error(self):
        """Test that dimension mismatch raises error."""
        query = [[1.0, 0.0]]
        doc = [[0.9, 0.1, 0.0]]  # Different dimension
        with pytest.raises(ValueError, match="Dimension mismatch"):
            rank_refine.maxsim_vecs(query, doc)

    def test_empty_tokens(self):
        """Test with empty token lists."""
        score = rank_refine.maxsim_vecs([], [])
        assert isinstance(score, float)

    def test_single_token(self):
        """Test with single token."""
        query = [[1.0, 0.0]]
        doc = [[0.9, 0.1]]
        score = rank_refine.maxsim_vecs(query, doc)
        assert isinstance(score, float)

