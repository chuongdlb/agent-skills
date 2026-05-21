---
name: tex-source-paper-extractor
description: >
  Extract or enrich a paper card from an arXiv paper's LaTeX e-print source (.tex) instead of its PDF. Use for any arXiv paper — TeX preserves formulas exactly, exposes tables/algorithms/appendices cleanly, and avoids PDF math-extraction loss.
layer: meta
domain: [general]
source-project: papers-kb
depends-on: [paper-extractor]
tags: [paper, extraction, latex, tex, arxiv, knowledge-base, enrichment]
---

# TeX-Source Paper Extractor

## Purpose

Produce or enrich a structured "paper card" by reading a paper's **arXiv LaTeX e-print source** rather than its compiled PDF. The card format and evaluative-extraction philosophy are identical to `paper-extractor` — this skill only changes the *input pathway*. TeX source is preferred for arXiv papers because:

- Formulas are already in LaTeX (no lossy PDF math OCR) — satisfies the KB rule that all formulas stay as `$inline$` / `$$display$$`.
- Tables, algorithms, and **appendices** (hyperparameters, ablations, proofs) are cleanly accessible — these are exactly the parts PDF/metadata extraction tends to drop.
- The e-print is often a **newer/extended version** (e.g. journal/IJRR) than the originally-cited venue.

## When to Use

- Creating a new card for a paper that is on arXiv.
- **Enriching** an existing thin or PDF-built card for an arXiv paper (the common case — see the `tex-enrichment-batches.md` initiative in the papers KB).
- Any time formula fidelity matters.

**Not for:** non-arXiv papers (fall back to `paper-extractor` on the PDF), lossless book extraction (`book-to-knowledge-base`), or topic synthesis (`kb-integrator`).

## Inputs

- arXiv ID (e.g. `2303.04137`) and the target card id/path.
- KB taxonomy at `kb/config/taxonomy.md` for domain tags.

## Procedure

### 1. Download the e-print source
```bash
curl -sL -A "Mozilla/5.0 (research KB)" "https://arxiv.org/e-print/<ARXIV_ID>" -o src.tar.gz
tar xzf src.tar.gz 2>/dev/null || gunzip -c src.tar.gz > main.tex   # some are a single gzipped .tex
```
The endpoint returns a gzipped tarball (or, rarely, a single gzipped `.tex`).

### 2. Find the main file and follow includes
```bash
grep -rl "\\begin{document}" --include="*.tex" .
```
Multi-file papers use `\input{...}` / `\include{...}` — read the included section files (often under `sections/`, `text/`). Appendices and supplementary (`supp.tex`, `appendix.tex`) carry the highest-value extra content.

### 3. Extract following the `paper-extractor` card format
Use the same card template (frontmatter + One-Line Summary, Problem, Contributions, Method Summary, Key Results table, Baselines, Limitations, Novelty Claims, Relevance). Then specifically harvest from TeX what PDFs lose:
- Exact equations (copy LaTeX verbatim).
- Hyperparameter tables and training details (often appendix).
- Ablation results and secondary findings.
- Theoretical derivations / connections (e.g. control-theory limits, EBM/score-function arguments).

### 4. Resolve custom LaTeX macros — CRITICAL
Papers define private macros (`\newcommand{\obs}{...}`, `\vox`, `\shortname`, `\ours`, `\qattn`, …). These will **not render** in MathJax/markdown. Before writing the card:
- Check `\newcommand` / `\def` definitions (often in `macros.tex`, `notation.tex`, or the preamble).
- Replace each macro with standard renderable notation (e.g. `\obs → o`, `\vox → V`).
- Verify no leftover macros: `grep -E '\\(obs|vox|shortname|ours|qattn)\b' <card>`.

### 5. Verify metadata against the source
Stub/old cards frequently have wrong metadata. Fix from the TeX `\author{}`, title, and any venue macro:
- `authors: ["Various"]` → real author list.
- Wrong/placeholder venue (e.g. arXiv-format default) → correct venue.
- **Version drift:** if the e-print is an extended/journal version (look for journal class files like `iclr2025_conference`, `IEEEtran`, `ijrr` figure prefixes, `\iclrfinalcopy`), do not silently attribute later-version content to the originally-cited venue. Note the mismatch.

### 6. Save and quality-check
Write to `kb/papers/<id>.md`. Run the `paper-extractor` quality checklist plus: all formulas LaTeX, no unresolved macros, metadata matches source.

## Batch / token-efficiency mode (gap scan)

When enriching many already-rich cards, don't read every full TeX into context. Use a mechanical gap-scan first to rank which cards actually have missing content (high word-ratio is *expected* and not itself a defect — look for appendices, many equations/tables, and substantive uncovered section titles). The reusable script and the 13-batch plan live in the papers KB at `kb/reports/tex-enrichment-batches.md`.

## Common Pitfalls

1. **Leftover macros** rendering as raw `\obs` in the card — always resolve (step 4).
2. **Missing `\input` files** — the main `.tex` may be near-empty; the content is in included section files.
3. **Treating the e-print as the cited version** — it may be newer; check class files (step 5).
4. **Over-enriching** — cards are evaluative summaries, not transcriptions; pull what's KB-relevant (method/results/theory), skip acknowledgements, author lists, reproducibility boilerplate.
