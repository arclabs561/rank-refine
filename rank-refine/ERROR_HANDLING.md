# Error Handling Design

This document explains the error handling philosophy and patterns used in `rank-refine`.

## Philosophy

`rank-refine` uses a **Result-based error handling** approach:

- **Most functions return `Result<T, RefineError>`** for explicit error handling
- **Invalid inputs return errors** rather than panicking or returning empty results
- **Validation happens at API boundaries** to catch errors early
- **Debug assertions** catch programming errors in development

## Why This Design?

1. **Safety**: Refinement operations can fail in meaningful ways (dimension mismatches, empty inputs)
2. **Clarity**: Errors are explicit, not hidden in empty results
3. **Debugging**: Error messages help identify issues (e.g., "dimension mismatch: query 128, doc 256")
4. **Consistency**: All refinement functions follow the same pattern

## Error Handling Patterns

### Pattern 1: Result for All Operations

```rust
pub fn pool_tokens(
    tokens: &[Vec<f32>],
    pool_factor: usize,
) -> Result<Vec<Vec<f32>>, RefineError> {
    if tokens.is_empty() {
        return Err(RefineError::EmptyInput);
    }
    if pool_factor == 0 {
        return Err(RefineError::InvalidParameter("pool_factor must be > 0"));
    }
    // ... pooling logic
}
```

**When to use**: All public functions that can fail.

**Examples**: `pool_tokens`, `pool_tokens_adaptive`, `refine`, `normalize`

### Pattern 2: Debug Assertions

```rust
pub fn maxsim(
    query: &[&[f32]],
    doc: &[&[f32]],
) -> f32 {
    #[cfg(debug_assertions)]
    {
        // Validate dimensions in debug builds
        for q in query {
            for d in doc {
                debug_assert_eq!(q.len(), d.len(), "dimension mismatch");
            }
        }
    }
    // ... maxsim logic
}
```

**When to use**: Internal functions where performance matters, but we want safety in debug builds.

**Examples**: `maxsim`, `cosine`, `dot` (SIMD functions)

### Pattern 3: Panic for Programming Errors

```rust
pub fn mmr<I: Clone>(
    candidates: &[(I, f32)],
    similarity: &[f32],
    config: MmrConfig,
) -> Vec<(I, f32)> {
    try_mmr(candidates, similarity, config)
        .expect("similarity matrix must be n×n")
}
```

**When to use**: Convenience functions that wrap fallible versions, when the error indicates a programming bug.

**Examples**: `mmr` (wraps `try_mmr`), convenience wrappers

## Error Types

### RefineError

```rust
pub enum RefineError {
    /// Dimension mismatch between vectors.
    DimensionMismatch { expected: usize, actual: usize },
    /// Empty input where non-empty is required.
    EmptyInput,
    /// Invalid parameter value.
    InvalidParameter(String),
    /// Numerical error (NaN, Infinity, division by zero).
    NumericalError(String),
}
```

## Input Validation Guide

### What Gets Validated?

All public functions validate:
- ✅ Empty inputs (return `Err(RefineError::EmptyInput)`)
- ✅ Dimension mismatches (return `Err(RefineError::DimensionMismatch)`)
- ✅ Invalid parameters (return `Err(RefineError::InvalidParameter)`)
- ✅ Numerical errors (NaN, Infinity, division by zero)

### What Doesn't Get Validated?

- ❌ Non-normalized vectors (assumed to be normalized, may produce unexpected results)
- ❌ Very small vectors (may fail normalization, but that's expected)
- ❌ Similarity matrix correctness (validated in `try_mmr`, panics in `mmr`)

## Recommended Error Handling Strategy

```rust
use rank_refine::colbert;
use rank_refine::RefineError;

// 1. Handle errors explicitly
match colbert::pool_tokens(&tokens, 2) {
    Ok(pooled) => {
        // Use pooled tokens
        process_tokens(&pooled)
    }
    Err(RefineError::EmptyInput) => {
        // Handle empty input
        eprintln!("Warning: empty tokens, skipping pooling");
        tokens.clone()
    }
    Err(RefineError::InvalidParameter(msg)) => {
        // Handle invalid parameter
        eprintln!("Error: {}", msg);
        return Err(format!("Invalid pool_factor: {}", msg));
    }
    Err(e) => {
        // Handle other errors
        return Err(format!("Pooling failed: {:?}", e));
    }
}
```

## Edge Case Behavior

| Input | Behavior | Error Type |
|-------|----------|------------|
| Empty tokens | Returns `Err(RefineError::EmptyInput)` | ❌ Error |
| pool_factor=0 | Returns `Err(RefineError::InvalidParameter)` | ❌ Error |
| Dimension mismatch | Returns `Err(RefineError::DimensionMismatch)` | ❌ Error |
| Zero vector normalization | Returns `None` (Option, not Result) | ⚠️ None |
| Very small vector | May return `None` (near-zero magnitude) | ⚠️ None |
| Similarity matrix wrong size | Panics in `mmr`, returns `Err` in `try_mmr` | ❌ Error |

## Comparison with rank-fusion

| Aspect | rank-fusion | rank-refine |
|--------|-------------|-------------|
| Default return type | `Vec<T>` | `Result<T, E>` |
| Error handling | Graceful degradation | Explicit errors |
| Validation | Opt-in (`validate()`) | Built-in (Result) |
| Empty inputs | Returns empty Vec | Returns error |
| Philosophy | Simplicity, performance | Safety, clarity |

## Summary

- **Design**: Result-based, explicit error handling
- **All public functions**: Return `Result<T, RefineError>`
- **Validation**: Built-in at API boundaries
- **Debug builds**: Additional assertions for safety
- **Production**: Always handle errors explicitly

