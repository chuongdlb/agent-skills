---
name: paper-extractor
description: >
  Extract structured contributions from a research paper PDF into a paper card with metadata, method summary, key results, and novelty claims.
layer: meta
domain: [general]
source-project: papers-kb
depends-on: []
tags: [paper, extraction, knowledge-base, pdf, research]
---

# Paper Extractor — Structured Paper Card Generation

## Purpose

Read one research paper PDF and produce a structured "paper card" (~80-150 lines) that captures the paper's contributions, methods, results, and novelty claims. This is **evaluative extraction** — not lossless transcription. Focus on what's new and how it compares to prior work.

## When to Use

Invoke this skill when:
- You have a research paper PDF to add to the knowledge base
- You need a structured summary for comparison and integration
- You are processing papers as part of a `kb-maintenance` cycle

**Not for:** Lossless book extraction (use `book-to-knowledge-base`), or topic-level synthesis (use `kb-integrator`).

## Input

- PDF file path (in `pdf/` directory)
- The KB taxonomy file at `kb/config/taxonomy.md` (for consistent domain tagging)

## Output

A paper card file at `kb/papers/<first-author-lastname><year>-<keyword>.md`

### File Naming Convention

- `<first-author-lastname>` — lowercase, no diacritics
- `<year>` — 4-digit publication year
- `<keyword>` — 1-2 word slug from title (lowercase, hyphenated)
- Examples: `kong2022-marsim.md`, `xu2024-omnidrones.md`, `song2023-autonomous-racing.md`

## Paper Card Format

```markdown
---
title: "<Full Paper Title>"
authors: [First Author, Second Author, ...]
year: YYYY
venue: "Conference/Journal Name"
doi: "10.xxxx/xxxxx"
arxiv: "XXXX.XXXXX"
pdf: "pdf/<filename>.pdf"
type: survey | system | method | application | benchmark | theory
domains: [domain-tag-1, domain-tag-2]
novelty: pending
---

# <Short Title or Acronym>

## One-Line Summary

<Single sentence: what this paper does that hasn't been done before.>

## Problem Statement

<2-3 sentences: What gap does this paper address? Why does it matter?>

## Contributions

1. <Contribution 1 — specific and evaluable>
2. <Contribution 2>
3. <Contribution 3 (if applicable)>

## Method Summary

<5-15 lines describing the core technical approach. Include:>
- Architecture/algorithm overview
- Key design choices and why
- Training procedure (if ML-based)
- Key equations or formulations (only the most important 1-2)

## Key Results

| Benchmark/Task | Metric | This Paper | Previous Best | Improvement |
|---------------|--------|------------|---------------|-------------|
| ... | ... | ... | ... | ... |

<If no quantitative comparison, describe qualitative results in 3-5 lines.>

## Baselines Compared

- <Baseline 1>: <brief description>
- <Baseline 2>: <brief description>

## Limitations

- <Limitation 1 — as stated by authors or identified from methodology>
- <Limitation 2>

## Novelty Claims

<What the authors claim is new. Be specific — not "we propose a new method" but "first to combine X with Y for task Z".>

## Key References

- <Ref 1>: <why it's important to this paper>
- <Ref 2>: <why it's important>
- <Ref 3>: <why it's important>

## Relevance to KB Topics

- <topic-tag-1>: <how this paper relates>
- <topic-tag-2>: <how this paper relates>
```

## Extraction Protocol

### Step 1: Read the Paper

Read the PDF using the Read tool with page ranges. For most papers (8-15 pages):
- Pages 1-5: title, abstract, intro, related work
- Pages 5-10: methodology
- Pages 10-end: experiments, results, conclusion

For longer papers (>15 pages), add intermediate ranges.

### Step 2: Extract Metadata

From the first page and references:
- Title, authors, year, venue
- DOI and/or arXiv ID (check footer, header, or references)
- Determine paper type from taxonomy

### Step 3: Evaluate Contributions

Read the introduction's contribution list and the conclusion. Cross-reference with the methodology and results sections. For each claimed contribution:
- Is it actually novel, or a known technique applied to a new domain?
- Is it supported by experimental results?
- How does it compare to cited baselines?

### Step 4: Summarize Method

Focus on architectural and algorithmic choices. Skip standard details (e.g., "we use Adam optimizer with lr=3e-4") unless they are part of the contribution. Include the 1-2 most important equations.

**CRITICAL: Preserve all formulas as LaTeX.** Any mathematical formula encountered in the PDF must be rendered using LaTeX notation: `$inline$` for inline math and `$$display$$` for display equations. Never describe formulas in plain text — always convert to proper LaTeX blocks.

### Step 5: Extract Results

Build the results table from the experiments section. Include:
- The benchmarks/tasks evaluated
- Metrics used
- This paper's results vs. previous best
- Compute improvement percentages where possible

### Step 6: Identify Limitations

Check the paper's limitations section (if present) and add any methodological limitations you identify (e.g., simulation-only, single drone type, narrow task).

### Step 7: Tag and Save

- Assign domain tags from `kb/config/taxonomy.md`
- Set `novelty: pending` (will be classified by `kb-integrator`)
- Write the card to `kb/papers/`

## Quality Checks

Before saving, verify:
- [ ] All YAML frontmatter fields are populated
- [ ] Contributions are specific and evaluable (not vague)
- [ ] Results table has actual numbers (not "we outperform")
- [ ] At least 2 limitations identified
- [ ] Domain tags match taxonomy vocabulary
- [ ] File name follows naming convention

## Common Pitfalls

1. **Vague contributions**: "We propose a novel framework" → Replace with specifics
2. **Missing baselines**: If the paper doesn't compare to baselines, note this as a limitation
3. **Overclaiming novelty**: Check if the "novel" technique exists in cited references
4. **Ignoring supplementary**: Some key results are in appendices — check if they exist
