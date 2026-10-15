---
name: kb-survey
description: >
  Produce a quantified survey of a research trend or topic from the knowledge base: scope a slice of cards (strand, query or id list), generate the tables, figures and bibliography deterministically, then write the survey prose from them with every number and claim traceable to a card.
layer: meta
domain: [general]
source-project: papers-kb
depends-on: [kb-query]
tags: [knowledge-base, survey, related-work, quantification, visualization, bibliography, strands]
---

# KB Survey — Quantified Landscape of a Topic

## Purpose

Turn a slice of the KB into a survey draft whose numbers, figures and bibliography are generated, and whose prose an agent writes from those artefacts. The output is a scaffold-backed survey, not a literature survey: it covers what the KB holds, and says so.

**Use for:** "survey the work on X", "quantify the trend in Y", a related-work landscape for a paper strand, a comparison table over a topic, a reading-list-with-numbers before a scout.

**Not for:** discovering papers outside the KB (use `paper-discoverer`), answering one question (use `kb-query`), or editing cards.

## Inputs you must settle with the user first

1. **The slice.** One of: a strand id from `kb/config/strands.json` (`--strand nd2-p1`), a natural-language query (`--query "..." -k 60`, add `--hybrid` for ND1/ND3/benchmark topics, keep plain BM25 for ND2 sync/AoI and OJ-VT topics), or an id list (`--ids-file`). Optional filters: `--topics a,b`, `--since YEAR`, `--core`.
2. **The audience and length.** A section for a paper (1–2 pages, comparison table mandatory) or a standalone report (5–10 pages, all sections).
3. **The name** of the output folder.

If the slice is under 10 cards, say so and ask whether to widen it; a quantified survey of 6 cards is a table, not a survey.

## Procedure

### Step 1: Generate the scaffold

```bash
uv run --with matplotlib scripts/kb_survey.py --strand <id> [--name <slug>]
uv run --with matplotlib scripts/kb_survey.py --query "<text>" -k 60 --core [--hybrid]
```

This writes `kb/reports/surveys/<name>-<date>/` with `slice.jsonl`, `tables.md`, `refs.bib`, `figures/*.svg` (`--png` for PNG) and `scaffold.md`. Read `tables.md` in full and `scaffold.md`. Do not recompute anything by hand.

### Step 2: Read the cards that carry the argument

- The 10 cards the scaffold lists for the comparison table (most cited, then `sota`): read each full card.
- One card per keyword family beyond those: read the One-Line Summary and Contributions.
- Every card in the slice: `tldr` from `slice.jsonl` is enough.
- Thin cards (listed in `tables.md`) are verified against their PDF or marked "thin card, unverified" when cited.

### Step 3: Write `survey.md` next to the scaffold

Keep the scaffold's eight sections. Rules for each:

1. **Scope and method.** How the slice was built (copy the selection line), its size, its years, and the limit: the slice is the KB, not the literature. Name the strands it touches.
2. **Quantified landscape.** Embed the four figures. One paragraph per figure that reads the numbers from `tables.md` (counts by year, venue mix, top keywords, co-occurring pairs). No adjectives without a count behind them.
3. **Taxonomy of approaches.** Start from the keyword families; merge, split or rename them by what the cards actually do, then one paragraph per family citing cards as `[[id]]`. State the grouping rule you ended with.
4. **Lineage and timeline.** The sota/foundational timeline from `tables.md`, plus the `ancestors` edges; name what each later work changed relative to its ancestor, from the cards' Contributions.
5. **State of the art.** Quote the topic files' claims, then verify each against the card it names; drop or correct what the card does not support, and say which topic file the claim came from.
6. **Comparison table.** Fill the scaffold table by reading the 10 cards. Every cell comes from the card, cited as `[[id]]` with the section name; a cell you cannot support is "—", never a guess.
7. **Open problems and gaps.** Thin cards, strands never scouted (from `kb/reports/strand-coverage.md`), the coverage-gap lists in `kb/candidates/`, and the ancestors cited from outside the slice. End with the three questions the slice cannot answer.
8. **Bibliography.** Cite with the `refs.bib` keys (card ids). For a thesis manuscript, run `uv run scripts/kb_bib.py diff <manuscript.bib> --strand <id>` and report disagreements in the hand-back.

### Step 4: Hand back

Report the folder path, the slice size and selection, the three strongest quantified findings, every card cited that is thin or whose abstract looks wrong, and the coverage limits. Offer the `kb_bib.py export --keys authoryear` form if the user's manuscript uses author-year keys.

## Rules

- Every number comes from `tables.md`; every claim cites a card id; every comparison-table cell names the card section it came from.
- Never present the slice as the field. Say "of the N cards in the KB slice".
- Figures: one series per chart, the reference palette the script uses, no dual axes, no rainbow; if you add a chart, follow the `dataviz` skill and keep the table view in `tables.md`.
- Do not edit cards, topic files or the registry from this skill; file card defects you notice in the hand-back for `kb-lint`.
- Re-running the script overwrites the generated files but never `survey.md`.
