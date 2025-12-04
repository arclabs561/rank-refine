"""Type stubs for rank-refine Python bindings.

This file provides static type checking support for mypy, pyright, and other type checkers.
"""

from typing import List, Tuple, Optional

# Type aliases
Vector = List[float]
TokenEmbeddings = List[Vector]
RankedList = List[Tuple[str, float]]
Embeddings = List[Vector]
SimilarityMatrix = List[float]
Alignment = Tuple[int, int, float]


# Configuration Classes
class MmrConfigPy:
    """Configuration for Maximal Marginal Relevance (MMR)."""
    
    lambda: float
    k: int
    
    def __init__(self, lambda: float = 0.5, k: int = 10) -> None: ...
    def with_lambda(self, lambda: float) -> "MmrConfigPy": ...
    def with_k(self, k: int) -> "MmrConfigPy": ...


class DppConfigPy:
    """Configuration for Determinantal Point Process (DPP)."""
    
    alpha: float
    k: int
    
    def __init__(self, alpha: float = 1.0, k: int = 10) -> None: ...
    def with_alpha(self, alpha: float) -> "DppConfigPy": ...
    def with_k(self, k: int) -> "DppConfigPy": ...


# Core SIMD Functions
def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity between two vectors."""
    ...


def dot(a: Vector, b: Vector) -> float:
    """Dot product of two vectors."""
    ...


def norm(v: Vector) -> float:
    """L2 norm of a vector."""
    ...


# MaxSim Functions
def maxsim_vecs(query_tokens: TokenEmbeddings, doc_tokens: TokenEmbeddings) -> float:
    """MaxSim (late interaction) scoring for token-level embeddings."""
    ...


def maxsim_cosine_vecs(query_tokens: TokenEmbeddings, doc_tokens: TokenEmbeddings) -> float:
    """MaxSim with cosine similarity variant."""
    ...


def maxsim_batch(query_tokens: TokenEmbeddings, doc_tokens_list: List[TokenEmbeddings]) -> List[float]:
    """Batch MaxSim scoring for multiple documents."""
    ...


# Alignment Functions
def maxsim_alignments_vecs(query_tokens: TokenEmbeddings, doc_tokens: TokenEmbeddings) -> List[Alignment]:
    """Token-level alignments for MaxSim."""
    ...


def maxsim_alignments_cosine_vecs(query_tokens: TokenEmbeddings, doc_tokens: TokenEmbeddings) -> List[Alignment]:
    """Token-level alignments with cosine similarity."""
    ...


def highlight_matches_vecs(query_tokens: TokenEmbeddings, doc_tokens: TokenEmbeddings, threshold: float) -> List[int]:
    """Highlighted document token indices above threshold."""
    ...


# Diversity Functions
def mmr(candidates: RankedList, similarity: SimilarityMatrix, lambda: float = 0.5, k: int = 10) -> RankedList:
    """Maximal Marginal Relevance (MMR) with precomputed similarity matrix."""
    ...


def mmr_cosine(candidates: RankedList, embeddings: Embeddings, lambda: float = 0.5, k: int = 10) -> RankedList:
    """MMR with cosine similarity computed from embeddings."""
    ...


def dpp(candidates: RankedList, embeddings: Embeddings, alpha: float = 1.0, k: int = 10) -> RankedList:
    """Determinantal Point Process (DPP) diversity selection."""
    ...


# ColBERT Functions
def pool_tokens(tokens: TokenEmbeddings, pool_factor: int) -> TokenEmbeddings:
    """Pool tokens to reduce storage (clustering-based)."""
    ...


def colbert_alignments(query: TokenEmbeddings, doc: TokenEmbeddings) -> List[Alignment]:
    """ColBERT token alignments (convenience wrapper)."""
    ...


def colbert_highlight(query: TokenEmbeddings, doc: TokenEmbeddings, threshold: float) -> List[int]:
    """ColBERT highlight (convenience wrapper)."""
    ...


# Matryoshka Functions
def matryoshka_try_refine(
    candidates: RankedList,
    query: Vector,
    docs: List[Tuple[str, Vector]],
    head_dims: int,
    alpha: float = 0.5,
) -> RankedList:
    """Matryoshka refinement (fallible version to avoid panics)."""
    ...

