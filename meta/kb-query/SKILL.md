---
name: kb-query
description: >
  Answer research questions against the knowledge base using ranked BM25 search, curated topic files, paper cards, and structured registry filters.
layer: meta
domain: [general]
source-project: papers-kb
depends-on: [paper-extractor, kb-integrator]
tags: [knowledge-base, query, research, sota-lookup, comparison, bm25, retrieval, ranked-search]
---

# KB Query — Research Question Answering

## Purpose

Answer research questions by querying the knowledge base. This is a **read-only** skill — it searches ranked indexes, topic files, paper cards, and the registry but never modifies them. The only write it performs is appending to the query log (see Query Logging).

## When to Use

Invoke this skill when the user asks:
- "What's the state of the art in X?"
- "Compare paper A vs paper B"
- "What are the open problems in X?"
- "Which papers use method Y?"
- "What's the timeline of progress in X?"
- Any free-form research question against the KB

**Not for:** Adding papers (use `paper-extractor` + `kb-integrator`), or discovering new papers (use `paper-discoverer`).

## Retrieval Protocol — always in this order

1. **Ranked search first** for any topical question:
   ```bash
   uv run scripts/kb_search.py "<the question, natural language>" -k 20 --json
   ```
   Field-weighted BM25 (title 6× / topics 5× / tldr 3× / body 1×) with acronym-alias expansion from `kb/config/aliases.json`. Measured on the gold set: Recall@20 ≈ 0.80 vs 0.06–0.14 for grep funnels — ranked search is the default entry point, not grep.

2. **Structured filters** for exact facets (novelty, code, type, topics, year):
   ```bash
   jq -c 'select(.novelty == "sota" and (.topics | index("digital-twin-uav")))' kb/compact.jsonl
   jq -c 'select(.code != null and (.topics | index("rl-for-flight")))' kb/compact.jsonl
   ```
   If `jq` is not installed on the host, filter with Python instead:
   ```bash
   uv run python -c "import json;[print(r['id'],r['novelty']) for r in map(json.loads,open('kb/compact.jsonl')) if 'digital-twin-uav' in r['topics']]"
   ```

3. **Escalate by tier, never bulk-load**: ranked hits → `kb/compact.jsonl` rows (tldr, novelty, venue) for the candidates → full cards `kb/papers/<id>.md` for the **3–5 finalists only**.

4. **Curated topic files outrank search for SoTA questions**: `kb/topics/<category>/<topic>.md` "State of the Art" sections are maintained by `kb-integrator`. Resolve topic names via `kb/topics/_index.md` (topics live in 7 category directories — there are no flat `kb/topics/<topic>.md` files). Cross-check the topic file's SoTA claim with one ranked search to catch anything newer than the last integration.

5. **Grep is the fallback, not the default** — use it for exact strings (a method name, a benchmark name, an author) or when ranked search returns nothing relevant:
   ```bash
   grep -rlEi "exact-method-name" kb/papers/
   ```

## Query Types

### 1. SoTA Lookup — "What's the state of the art in <topic>?"
1. Map the question to a topic via `kb/topics/_index.md` (or `kb/config/taxonomy.md`)
2. Read the topic file's "State of the Art" section — this is the curated answer
3. Run one ranked search to verify nothing newer exists in the KB
4. Return the SoTA summary plus the results table from the SoTA paper's card

### 2. Paper Comparison — "Compare <A> vs <B>"
1. Resolve both card ids (ranked search if the user gave nicknames)
2. Read both cards from `kb/papers/`
3. Build a comparison table: method approach, benchmarks, key results side-by-side, strengths/limitations, novelty classification

### 3. Gap Analysis — "What are the open problems in <topic>?"
1. Read the topic file's "Open Problems" section
2. Pull the topic's surveys: `jq -c 'select(.type == "survey" and (.topics | index("<tag>")))' kb/compact.jsonl`
3. Cross-reference finalist cards' "Limitations" sections; check whether recent papers already address a listed problem (compare years)

### 4. Method Search — "Which papers use <method>?"
1. Ranked search with the method name (aliases expand acronyms automatically)
2. Grep the exact method string across `kb/papers/` to catch mentions the ranker down-weighted
3. Return papers with one line each on how they use the method

### 5. Timeline — "How did <area> evolve?"
1. Topic file for the curated narrative; `kb/timeline.md` for integration history
2. `jq` the topic's papers sorted by year; mark SoTA transitions explicitly

### 6. Coverage — "What does the KB cover on X?"
1. `kb/stats.md` for headline metrics; `jq` counts on `kb/compact.jsonl` for the specific slice

### 7. Free-Form
1. Ranked search → tier escalation (protocol above) → synthesize, citing specific cards

## Query Logging (feeds the retrieval gold set)

After answering a real user question, append one line to `kb/eval/query-log.jsonl`:
```json
{"date": "YYYY-MM-DD", "type": "<router type>", "query": "<user question verbatim>", "cited": ["id1", "id2"]}
```
This log is the raw material for expanding `kb/eval/queries.jsonl` (re-pooled and judged during maintenance — never edit `queries.jsonl` directly from a query session).

## Response Format

All responses should:
- Cite papers by card id (e.g., `kong2022-marsim`) and link `kb/papers/<id>.md`
- Include quantitative results where available; note each cited paper's novelty classification
- Flag KB coverage gaps relevant to the question and suggest a `paper-discoverer` query when a gap appears
- Respect the token funnel: never read more than ~5 full cards or bulk-load `kb/compact.jsonl` into context

## Examples

**"What's the best drone simulator?"** → `_index.md` maps to A-category simulation topics → read topic file SoTA → `uv run scripts/kb_search.py "GPU-parallel drone simulator RL training" -k 10` to verify freshness → answer with simulator name, key features, and how alternatives differ.

**"The Safety Gym paper?"** (known item) → `uv run scripts/kb_search.py "safety gym constrained RL benchmark" -k 5 --json` → top hit `ray2019-safety-gym-ppolag` → read that one card → answer.
