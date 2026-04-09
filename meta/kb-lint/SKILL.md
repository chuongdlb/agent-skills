---
name: kb-lint
description: >
  Health-check the KB: run mechanical checks (YAML, enums, broken links, orphans, duplicates)
  via Python script, then semantic checks (stale SoTA, missing cross-references, coverage gaps)
  via LLM. Produces a lint report filed into the wiki. Optional fix mode walks through findings interactively.
layer: meta
domain: [general]
source-project: papers-kb
depends-on: [kb-query]
tags: [knowledge-base, lint, health-check, quality, maintenance]
---

# KB Lint — Knowledge Base Health Check

## Purpose

Detect and surface data integrity issues, stale claims, broken cross-references, and improvement opportunities across the KB. Two layers:

1. **Mechanical checks** (Python script) — fast, deterministic, covers YAML/enum/link/orphan/duplicate validation
2. **Semantic checks** (LLM) — stale SoTA detection, missing cross-references, coverage gaps, suggested queries

## When to Use

- Periodically (weekly or after large ingests) as standalone: `/kb-lint`
- As part of `kb-maintenance` Step 5 (report-only mode)
- Before committing large KB changes to verify integrity
- When the user asks to "check KB health", "find issues", "lint the KB"

## Modes

### Report-only (default)

Produce the lint report without user interaction. Used by `kb-maintenance`.

### Fix mode

Invoked with `/kb-lint --fix` or "run kb-lint with fixes". Walks through findings interactively, proposing fixes for approval.

---

## Step 1: Run Mechanical Checks

Execute the Python script to get deterministic findings:

```bash
uv run scripts/kb_lint.py
```

This writes `kb/reports/lint-findings.json` with all mechanical findings.

### What the script checks

| ID | Check | Severity | Description |
|----|-------|----------|-------------|
| M1 | YAML frontmatter | error/warning | Required fields: title, year, type, topics/domains. Warning for missing novelty. |
| M2 | Enum validation | error | `type` must be: method, system, survey, benchmark, application, theory |
| M3 | Domain tag validation | error | Every domain tag must exist in `kb/config/taxonomy.md` |
| M4 | Registry-cards alignment | error | Every registry entry has a card; every card has a registry entry |
| M5 | Registry-index alignment | warning | Every registry entry has an index row and vice versa |
| M6 | Broken links in topic files | error | All `[id](../../papers/id.md)` links resolve to existing files |
| M7 | Duplicate detection | warning | Same arXiv ID or DOI under different paper IDs |
| M8 | Pending novelty | warning | Cards still marked `novelty: pending` |
| M9 | Thin SoTA cards | warning | SoTA cards with <15 lines or missing Key Results section |
| M10 | Stats drift | warning | `kb/stats.md` counts vs actual counts |

### Read the findings

After the script completes, read `kb/reports/lint-findings.json`:

```json
{
  "summary": { "error": N, "warning": M, "info": K },
  "findings": [ { "id": "M1-001", "check": "...", "severity": "...", "file": "...", "message": "...", "fixable": true } ]
}
```

---

## Step 2: Semantic Checks (LLM)

Run these checks using the mechanical findings as a starting point. Only read individual files when needed — don't scan all 1,300+ cards.

### S1: Stale SoTA Detection

For each of the 36 topic files:

1. Read the topic file
2. Find the "State of the Art" section — extract the paper ID, year, and key result
3. Scan the "Papers Contributing" table for papers with:
   - `novelty: sota` or `novelty: complementary` AND year > SoTA year
4. For each candidate: read the paper card's Key Results section
5. If the newer paper reports better results on the same benchmark/task, flag as stale SoTA

**Output format per finding:**
```
S1: Topic [topic-name] — SoTA lists [Paper A] ([year], [metric]) but [Paper B] ([year]) reports [better metric] on same benchmark
```

**Severity:** warning

### S2: Missing Cross-References

1. Read `kb/registry.json`
2. Build a co-occurrence matrix: for each pair of domain tags, count papers that share both tags
3. For the top 10 most co-occurring tag pairs:
   - Read both topic files
   - Check if their "Related Topics" sections reference each other
   - Check if papers appearing in both topics' "Papers Contributing" tables are listed in both
4. Flag pairs with high overlap but no cross-references

**Severity:** info

### S3: Topic Coverage Gaps

1. From `kb/registry.json`, count papers per domain tag by year
2. Flag topics with:
   - Fewer than 3 total papers
   - No papers from 2025 or later
3. For each gap, suggest a search query that could fill it (based on the topic description in `kb/config/taxonomy.md`)

**Severity:** info

### S4: Report-Worthy Queries

Based on all findings (mechanical + semantic), suggest 2-3 questions worth investigating and filing as new report pages. These should synthesize across topics — the kind of analysis that compounds in the wiki.

Examples:
- "Which methods achieve SoTA on multiple benchmarks across different topics?"
- "How do papers in [topic A] and [topic B] differ in their approach to [shared challenge]?"

**Severity:** info

---

## Step 3: Write Lint Report

Write the report to `kb/reports/lint-YYYY-MM-DD.md` using this format:

```markdown
# KB Lint Report — YYYY-MM-DD

## Summary

| Metric | Value |
|--------|-------|
| Papers scanned | N |
| Topic files scanned | 36 |
| Errors | N |
| Warnings | M |
| Info | K |

## Errors (must fix)

### M2: Invalid Type Values (N)
- `papers/foo2024-bar.md` — type 'journal' not in allowed enum
...

### M3: Unknown Domain Tags (N)
- `papers/baz2025-qux.md` — unknown tag 'digital-twin-networking'
...

[Continue for all error-level findings, grouped by check ID]

## Warnings (should fix)

### S1: Stale SoTA (N)
- **rl-for-flight** — SoTA lists Chen 2023 (87.3%) but Li 2025 reports 94.1% on same benchmark
...

### M7: Duplicate Papers (N)
- `paper-a.md` and `paper-b.md` share arXiv ID 2401.12345
...

### M9: Thin SoTA Cards (N)
- `papers/foo2024-bar.md` — SoTA card with only 8 non-empty lines
...

[Continue for all warning-level findings]

## Info (opportunities)

### S3: Coverage Gaps (N)
- **channel-modeling** — 4 papers, none from 2025+
  - Suggested query: "neural channel estimation 6G 2025"
...

### S4: Suggested Queries
- "How do VLA architectures compare to VLM+RL pipelines for manipulation?"
  → Would synthesize papers across F.3 and G.2, worth filing as a report
...
```

---

## Step 4: Fix Mode (interactive only)

**Skip this step in report-only mode (default).**

When invoked with fix mode:

1. Present the lint report summary to the user
2. Walk through findings by severity (errors first, then warnings, then info)
3. For each finding, offer a fix:

### Mechanical fixes (propose and apply on approval)

- **M2 invalid type**: Read the paper card, determine correct type from content, propose edit
- **M3 unknown tag**: Check if the tag is close to a valid tag (typo), or suggest the closest valid tag
- **M4 orphan card**: Offer to create a registry entry from the card's frontmatter
- **M5 missing index row**: Offer to add the row to index.md
- **M8 pending novelty**: Read the card and classify using the scoring rubric in `kb/config/scoring-rubric.md`
- **M9 thin SoTA card**: If PDF exists in `pdf/`, offer to re-extract using `paper-extractor`
- **M10 stats drift**: Offer to run `uv run scripts/update_kb_stats.py`

### Semantic fixes

- **S1 stale SoTA**: Read both papers, propose updated "State of the Art" section text for the topic file
- **S2 missing cross-refs**: Propose additions to "Related Topics" sections

### Info items

- **S3 coverage gaps**: Offer to run the suggested search query via `paper-discoverer`
- **S4 queries**: Offer to run the query via `kb-query` and file the result as a report page

### Fix tracking

Keep a running count of: fixes applied, skipped, deferred. After all findings are processed (or the user stops), append a fix summary to the lint report:

```markdown
## Fix Summary

- Fixes applied: N
- Fixes skipped: M
- Fixes deferred: K
- Remaining errors: E
- Remaining warnings: W
```

---

## Integration with kb-maintenance

When called from `kb-maintenance` Step 5, run in report-only mode:

1. Execute `uv run scripts/kb_lint.py`
2. Read `lint-findings.json`
3. Run semantic checks S1 and S2 (skip S3, S4 to save time)
4. Append findings summary to the cycle report
5. If errors > 0, warn the user before proceeding to git commit

---

## Important Notes

- **Never modify paper cards without user approval** in fix mode
- **S1 stale SoTA is approximate** — it compares years and looks for "better" results, but can't always parse benchmark tables. When in doubt, flag for human review rather than auto-fixing.
- **M9 only flags SoTA cards** — non-SoTA thin cards are by design (balanced dedup policy)
- **Dedup findings** — if the same paper triggers both M1 and M2, present them together in fix mode
