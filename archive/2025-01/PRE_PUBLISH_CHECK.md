# Pre-Publish Checklist

## ✅ GitHub Tags + Workflows
- [x] CI badges point to correct repos: `arclabs561/rank-refine` and `arclabs561/rank-fusion`
- [x] Workflow files exist: `.github/workflows/ci.yml` in both repos
- [x] Workflows are correct (test, msrv, clippy, fmt, docs)
- [x] Repository URLs in Cargo.toml match GitHub repos

## ✅ Documentation Quality
- [x] No fluffy/unnecessary verbosity (recently tightened)
- [x] READMEs are quintessential and correct
- [x] Cross-references between repos are accurate
- [x] All technical content is accurate

## ✅ Bugs
- [x] No TODO/FIXME/XXX/HACK/BUG markers (only Debug derives found, which are fine)
- [x] No obvious errors in documentation
- [x] Code compiles (cargo publish --dry-run would check this)

## ⚠️ Cruft
- [ ] `OUTREACH.md` - Contains outreach notes (not cruft, but not essential for users)
  - **Decision**: Keep for now (useful for maintainers, doesn't hurt)

## ❌ Not Pushed/Published
- [ ] **Uncommitted changes**:
  - `DESIGN.md` (modified)
  - `README.md` (modified)
  - `archive/2025-01/*` (new files)
  - `src/lib.rs` (rank-fusion only, modified)

- [ ] **Not pushed to GitHub**: Changes need to be committed and pushed
- [ ] **Not published to crates.io**: Versions are 0.7.35 (rank-refine) and 0.1.18 (rank-fusion)

## Action Items

1. **Commit changes**:
   ```bash
   git add DESIGN.md README.md archive/ src/lib.rs
   git commit -m "docs: improve documentation with motivation, visuals, and decision guides"
   ```

2. **Push to GitHub**:
   ```bash
   git push origin master
   ```

3. **Publish to crates.io** (if ready):
   ```bash
   cargo publish
   ```

## Notes
- Archive folder is tracked (good for history)
- OUTREACH.md is fine to keep (useful context)
- All documentation improvements are ready to commit

