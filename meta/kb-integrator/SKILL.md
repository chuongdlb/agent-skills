---
name: kb-integrator
description: >
  Compare a paper card against the existing knowledge base, classify its novelty, and update topic files, index, timeline, and registry.
layer: meta
domain: [general]
source-project: papers-kb
depends-on: [paper-extractor]
tags: [knowledge-base, integration, novelty-classification, topic-synthesis]
---

# KB Integrator — Novelty Classification and Topic Integration

## Purpose

Take one paper card (output of `paper-extractor`) and integrate it into the knowledge base by:
1. Classifying its novelty against existing KB content
2. Updating relevant topic files
3. Updating the index, timeline, and registry

**This skill must run sequentially** — it writes to shared topic files and the registry.

## When to Use

Invoke this skill when:
- A new paper card exists in `kb/papers/` with `novelty: pending`
- You are running a `kb-maintenance` cycle
- You are doing a cold start / bulk integration

**Not for:** Extracting papers (use `paper-extractor`), or querying the KB (use `kb-query`).

## Input

- Paper card path: `kb/papers/<name>.md`
- Existing KB: `kb/topics/`, `kb/index.md`, `kb/registry.json`
- Scoring rubric: `kb/config/scoring-rubric.md`

## Novelty Classification Algorithm

### Step 1: Load Context

1. Read the paper card's YAML frontmatter to get `domains` and `type`
2. Read `kb/config/scoring-rubric.md` for classification thresholds
3. Load relevant topic files based on domain tags (Glob `kb/topics/*.md`, filter by domain match)
4. Load `kb/registry.json` for dedup check

### Step 2: Dedup Check

Check `kb/registry.json` for:
- Matching DOI
- Matching arXiv ID
- Title similarity (exact lowercase match after stripping punctuation)

If duplicate found: skip integration, log to timeline as "Skipped (duplicate of <existing-id>)".

### Step 3: Classify Novelty

For each contribution in the paper card:

1. **Scan existing topic files** for SoTA entries on the same benchmark/task
2. **Compare results**: If the paper's results table shows >10% improvement over the current SoTA entry → `sota`
3. **Check for new capability**: If the paper demonstrates something no existing entry does → `sota`
4. **Check for incremental**: If <10% improvement on same benchmarks → `incremental`
5. **Check for complementary**: If different methodology or different sub-problem → `complementary`
6. **Check for derivative**: If applying known methods to new domain without modification → `derivative`
7. **Check for survey**: If paper type is `survey` → `survey`

Assign the **highest** classification across all contributions.

### Step 4: Update Paper Card

Edit the paper card's YAML frontmatter:
- Change `novelty: pending` → `novelty: <classification>`

### Step 5: Update Topic Files

For each domain tag in the paper card:

1. If topic file doesn't exist → create it from the template below
2. If paper is `sota`:
   - Add full write-up to the "State of the Art" section
   - Move the previous SoTA entry to a "Previous SoTA" sub-section (keep 2-3 lines)
   - Update the "Methods Landscape" table
   - Add to "Papers Contributing" list
3. If paper is `incremental`:
   - Add 2-3 line summary to "Papers Contributing" noting specific addition over SoTA
   - Update "Methods Landscape" table if it adds a new method
4. If paper is `complementary`:
   - Add to "Papers Contributing" with 3-5 line summary
   - Update "Methods Landscape" table
   - Consider adding to "Open Problems" if it identifies new gaps
5. If paper is `derivative`:
   - Add 1-2 line entry to "Papers Contributing"
6. If paper is `survey`:
   - Add entry to a "Surveys" section (create if needed)
   - Check if survey identifies open problems not in "Open Problems" → add them

### Step 6: Update Index

Add a row to `kb/index.md`:

```
| <id> | <first-author> | <year> | <title> | <type> | <domains> | <novelty> | [card](papers/<filename>.md) |
```

### Step 7: Update Timeline

Add entry to `kb/timeline.md`:

```markdown
## YYYY-MM-DD

- Added: <id> — <title> [<novelty>]
- Updated: <topic-file> — <description>
```

### Step 8: Update Registry

Add entry to `kb/registry.json`:

```json
{
  "id": "<first-author><year>-<keyword>",
  "title": "<title>",
  "doi": "<doi or null>",
  "arxiv": "<arxiv-id or null>",
  "title_hash": "<lowercase title stripped of punctuation>",
  "domains": ["domain1", "domain2"],
  "novelty": "<classification>",
  "added": "YYYY-MM-DD"
}
```

Increment `total_papers`.

### Step 9: Update Stats

Update `kb/stats.md` with new counts.

## Topic File Template

When creating a new topic file:

```markdown
# <Topic Name>

## State of the Art

<!-- SoTA papers get full write-up here -->

## Methods Landscape

| Method | Paper | Year | Approach | Key Strength | Key Limitation |
|--------|-------|------|----------|-------------|----------------|

## Open Problems

<!-- Identified gaps and unsolved challenges -->

## Surveys

<!-- Survey papers covering this topic -->

## Papers Contributing

| Paper | Year | Novelty | Summary |
|-------|------|---------|---------|
```

## Quality Checks

After integration, verify:
- [ ] Paper card `novelty` field updated from `pending`
- [ ] At least one topic file updated
- [ ] Index has new row
- [ ] Timeline has new entry
- [ ] Registry has new entry with correct field values
- [ ] Registry `total_papers` matches actual count
- [ ] No duplicate entries in registry

## Handling Edge Cases

### Paper spans multiple domains
Update ALL relevant topic files. The paper appears once in the index/registry but in multiple topic files.

### No existing topic file matches
Create a new topic file. Use the domain tag as the filename slug (e.g., `drone-simulation.md`).

### Paper supersedes existing SoTA
Move the old SoTA to "Previous SoTA" section. Keep 2-3 lines summarizing what it contributed. The new paper gets the full write-up.

### Cold start (empty KB)
The first paper in each domain automatically gets `sota` classification since there's nothing to compare against. Mark it as `sota (initial)` in the topic file.
