---
name: kb-query
description: >
  Answer research questions against the knowledge base using structured lookups across topic files, paper cards, and the index.
layer: meta
domain: [general]
source-project: papers-kb
depends-on: [paper-extractor, kb-integrator]
tags: [knowledge-base, query, research, sota-lookup, comparison]
---

# KB Query — Research Question Answering

## Purpose

Answer research questions by querying the knowledge base. This is a **read-only** skill — it searches across topic files, paper cards, the index, and the registry but never modifies them.

## When to Use

Invoke this skill when the user asks:
- "What's the state of the art in X?"
- "Compare paper A vs paper B"
- "What are the open problems in X?"
- "Which papers use method Y?"
- "What's the timeline of progress in X?"

**Not for:** Adding papers (use `paper-extractor` + `kb-integrator`), or discovering new papers (use `paper-discoverer`).

## Query Types

### 1. SoTA Lookup

**Question pattern**: "What's the state of the art in <topic>?"

**Procedure**:
1. Read `kb/config/taxonomy.md` to map the question to domain tags
2. Read the relevant topic file(s) in `kb/topics/`
3. Return the "State of the Art" section with the SoTA paper's full contribution summary
4. Include the results table from the SoTA paper's card

### 2. Paper Comparison

**Question pattern**: "Compare <paper A> vs <paper B>"

**Procedure**:
1. Read both paper cards from `kb/papers/`
2. Build a comparison table:
   - Method approach
   - Benchmarks evaluated
   - Key results (side-by-side where benchmarks overlap)
   - Strengths and limitations of each
3. Note which is classified as SoTA/incremental/etc.

### 3. Gap Analysis

**Question pattern**: "What are the open problems in <topic>?"

**Procedure**:
1. Read the relevant topic file(s)
2. Return the "Open Problems" section
3. Cross-reference with paper cards' "Limitations" sections for additional gaps
4. Check if any open problems have been addressed by recent papers (compare dates)

### 4. Method Search

**Question pattern**: "Which papers use <method/technique>?"

**Procedure**:
1. Grep across `kb/papers/*.md` for the method name in "Method Summary" sections
2. Grep across `kb/topics/*.md` in "Methods Landscape" tables
3. Return a list of papers with brief descriptions of how they use the method

### 5. Timeline Query

**Question pattern**: "What's the timeline of progress in <topic>?"

**Procedure**:
1. Read the relevant topic file
2. Read `kb/timeline.md` and filter entries for the topic's domain tags
3. Read paper cards sorted by year
4. Build a chronological narrative:
   - Year by year, what papers were published and what they contributed
   - Mark SoTA transitions explicitly

### 6. Coverage Query

**Question pattern**: "What does the KB cover?" / "How many papers do we have on X?"

**Procedure**:
1. Read `kb/stats.md` for high-level metrics
2. Read `kb/index.md` for the full paper list
3. Filter by domain tags if a specific topic is requested
4. Report: total papers, breakdown by domain, breakdown by novelty classification

### 7. Free-Form Query

For questions that don't match the above patterns:
1. Grep across all KB files (`kb/**/*.md`) for relevant keywords
2. Read the most relevant matches
3. Synthesize an answer citing specific papers and topic file sections

## Response Format

All responses should:
- Cite specific papers by their card ID (e.g., `kong2022-marsim`)
- Link to paper cards: `kb/papers/<id>.md`
- Include quantitative results where available
- Note the novelty classification of cited papers
- Flag any gaps in KB coverage relevant to the question

## Example Queries and Expected Responses

### Example: "What's the best drone simulator?"

1. Read `kb/topics/drone-simulation.md`
2. Find SoTA section → identify the leading simulator
3. Return: simulator name, key features, benchmark results, how it compares to alternatives
4. Note: "This is based on N papers in the KB covering drone simulation"

### Example: "Compare MARSIM and OmniDrones"

1. Read `kb/papers/kong2022-marsim.md` and `kb/papers/xu2024-omnidrones.md`
2. Compare: simulation type (point-realistic vs physics-based), supported features, performance metrics
3. Note: different use cases — MARSIM for lightweight testing, OmniDrones for RL training
