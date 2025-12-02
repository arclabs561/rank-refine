# Publishing Guide

This guide covers publishing `rank-refine` to:
- **crates.io** (Rust crate)
- **PyPI** (Python package)
- **npm** (WebAssembly package, if applicable)

## Prerequisites

1. **crates.io account**: https://crates.io
   - Get API token: https://crates.io/me
   - Add to GitHub secrets as `CRATES_IO_TOKEN`

2. **PyPI account**: https://pypi.org
   - Get API token: https://pypi.org/manage/account/token/
   - Add to GitHub secrets as `PYPI_API_TOKEN`

3. **npm account** (for WASM): https://www.npmjs.com
   - Get access token: https://www.npmjs.com/settings/{username}/tokens
   - Add to GitHub secrets as `NPM_TOKEN`

## Publishing Workflow

### Automated (Recommended)

1. **Create a GitHub release**:
   ```bash
   git tag v0.7.36
   git push origin v0.7.36
   ```
   Then create a release on GitHub with the same tag.

2. **GitHub Actions will automatically**:
   - Publish to crates.io
   - Publish to PyPI
   - (WASM to npm if configured)

### Manual Publishing

#### 1. Publish Rust Crate

```bash
cd rank-refine
cargo publish --token YOUR_CRATES_IO_TOKEN
```

#### 2. Publish Python Package

```bash
cd rank-refine-python
uv venv
source .venv/bin/activate
uv tool install maturin

# Test first
maturin build --uv

# Publish
maturin publish --uv --username __token__ --password YOUR_PYPI_TOKEN
```

#### 3. Publish WebAssembly (npm) - Optional

If you want to publish WASM bindings:

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build for npm
cd rank-refine
wasm-pack build --target nodejs --out-dir pkg

# Publish to npm
cd pkg
npm publish --access public
```

## Version Management

- Update version in:
  - `Cargo.toml` (workspace root)
  - `rank-refine/Cargo.toml`
  - `rank-refine-python/pyproject.toml`
  - `rank-refine-python/Cargo.toml`

- Use semantic versioning:
  - `MAJOR.MINOR.PATCH`
  - Breaking changes: increment MAJOR
  - New features: increment MINOR
  - Bug fixes: increment PATCH

## Pre-Publish Checklist

- [ ] All tests pass: `cargo test --workspace`
- [ ] Clippy clean: `cargo clippy --workspace --all-features -- -D warnings`
- [ ] Formatted: `cargo fmt --check --all`
- [ ] Documentation builds: `cargo doc --workspace --no-deps`
- [ ] Version numbers updated in all files
- [ ] CHANGELOG.md updated
- [ ] README.md is up to date
- [ ] Python bindings tested: `maturin develop --uv` and import works

## Post-Publish

After publishing, verify:

1. **crates.io**: https://crates.io/crates/rank-refine
2. **PyPI**: https://pypi.org/project/rank-refine/
3. **npm** (if applicable): https://www.npmjs.com/package/@arclabs561/rank-refine

Test installation:

```bash
# Rust
cargo add rank-refine

# Python
pip install rank-refine

# npm (if applicable)
npm install @arclabs561/rank-refine
```

