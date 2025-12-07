# Shared Cursor Rules Template

This directory contains a **template** for shared Cursor rules that should be copied to each rank-* repository.

## How to Use

### For New Repositories

1. Copy this entire `.cursor/rules/` directory to your new repository:
   ```bash
   cp -r rank-rank/.cursor rank-fusion/.cursor
   ```

2. Add repository-specific rules in `.cursor/rules/repo-specific.mdc`

### For Existing Repositories

1. Copy `shared-base.mdc` to your repo's `.cursor/rules/` directory
2. Reference it in your repo-specific rules if needed

## Important Note

**Cursor does NOT automatically discover rules from shared directories.** Each repository must have its own `.cursor/rules/` directory for Cursor to discover and apply rules.

The `rank-rank/.cursor/rules/` directory serves as a **template/reference**, not an active configuration that Cursor will use automatically.

## Rule File Format

Rules use the `.mdc` format with frontmatter:

```markdown
---
title: "Rule Title"
id: rule-id
description: "What this rule does"
priority: 100
alwaysApply: true
globs: "**/*.rs"  # Optional: auto-attach for matching files
---

# Rule content here
```

## Best Practices

- Keep rules focused (max ~500 lines per file)
- Use `alwaysApply: true` for base rules
- Use `globs` for file-type-specific rules
- Document why rules exist
- Update shared template when adding new common rules

