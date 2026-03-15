---
name: publication-scout
description: >
  Discover publishable research sweet spots at keyword intersections via deep research, verify gaps against real literature, and output exactly 3 ISI/Scopus Q1/Q2 publication approaches with 5-Gates critical review.
---

# Publication Scout — Research Gap Discovery & Publication Approach Generator

## Purpose

Given a set of research topic keywords, systematically discover publishable research gaps at keyword intersections, verify them against current literature via deep research, and output exactly **3 publication approaches** targeting ISI/Scopus Q1/Q2 journals — each evaluated with the 5-Gates critical thinking protocol.

## When to Use

Invoke this skill when:
- Exploring a new research direction and need to find publishable sweet spots
- Preparing a research plan and need verified, gap-backed publication targets
- Want to evaluate whether a research idea is genuinely novel before investing effort
- Need structured pros/cons analysis of competing research directions

**Not for:** Background gathering on candidates/supervisors (do that beforehand). Not for writing full thesis plans (use the output as input to thesis planning). Not for paper extraction or KB updates (use `paper-extractor`, `kb-integrator`).

## Input

**Required:**
- `keywords`: 3-6 research topic keywords (e.g., "UAV", "digital twin", "MARL", "6G", "VLM")

**Optional:**
- `target_venues`: preferred journal/conference venues (e.g., "IEEE JSAC", "IEEE IoT Journal")
- `exclude`: topics or methods to avoid (e.g., "signal processing", "hardware design")
- `context`: brief description of researcher's strengths or constraints (1-2 sentences)
- `--skip-verification`: skip the deep-research gap verification step (not recommended)

## Output

- `docs/publication-scout/YYYY-MM-DD-<topic-slug>-approaches.md` — structured report with exactly 3 approaches
- Console summary with approach comparison table

## Pipeline

### Step 1: Keyword Combination & Landscape Mapping

**Goal:** Generate meaningful research intersections from the input keywords and map the current landscape at each intersection.

**Procedure:**

1. Generate all 2-keyword and 3-keyword combinations from input keywords
2. For each combination, query the local KB (if available):
   - Use `kb-query` skill pattern: grep across `kb/topics/**/*.md` and `kb/papers/*.md`
   - Collect: existing papers, methods landscape, open problems, SoTA
3. For each combination, run external searches:
   - **Semantic Scholar API**: `GET https://api.semanticscholar.org/graph/v1/paper/search?query=<combo>&limit=15&fields=title,authors,year,venue,abstract,citationCount`
   - **arXiv API**: `GET http://export.arxiv.org/api/query?search_query=all:<combo>&max_results=10&sortBy=submittedDate&sortOrder=descending`
4. Build a **landscape matrix**:

```
| Intersection | KB Papers | Recent Papers (2024-2026) | Identified Gaps | Saturation |
|-------------|-----------|---------------------------|-----------------|------------|
| UAV + DT    | 8         | 25+                       | 3               | HIGH       |
| MARL + DT sync | 0      | 3                         | 2               | LOW        |
| VLM + DTN + UAV | 0     | 0                         | 1               | VERY LOW   |
```

5. Rank intersections by **publishability score**:
   - LOW saturation (few papers) + HIGH relevance (matches keywords) = high score
   - Prefer intersections where gaps are specific and addressable

### Step 2: Gap Identification

**Goal:** Extract specific, concrete research gaps from the landscape analysis.

For each promising intersection (top 5-8 by publishability score):

1. Analyze existing papers' **limitations sections** (from KB cards or abstracts)
2. Cross-reference with **open problems** from KB topic files
3. Look for recurring patterns:
   - "Single-agent where multi-agent is needed"
   - "Simulation-only, no real-world validation"
   - "Domain A solved, but not adapted to Domain B"
   - "Text-only where multimodal is needed"
   - "Centralized where distributed is needed"
   - "Theoretical framework without implementation"
4. Formulate each gap as a precise claim:
   - **BAD:** "No work on X" (too broad, likely false)
   - **GOOD:** "Existing work on X uses single-agent DRL (Zhang 2024); no decentralized multi-agent formulation exists for the Y-specific setting"
5. Collect 6-10 candidate gaps

### Step 3: Gap Verification via Deep Research

> **DEFAULT BEHAVIOR: Always ask the user before running verification.**
>
> Prompt: "I've identified N candidate gaps. Deep research verification will launch parallel agents to check each gap against current literature (2024-2026). This takes 3-5 minutes but prevents false novelty claims. **Verify gaps now? [Y/n]**"
>
> - If user confirms (or presses enter): proceed with verification
> - If user skips: proceed to Step 4 but mark ALL gaps as `[UNVERIFIED]` in the output

**Verification procedure:**

For each candidate gap, launch a background research agent with:

1. **Search queries** — 4-6 specific queries designed to find counterevidence:
   - Direct keyword match: `"<method A>" AND "<domain B>" AND "<specific claim>"`
   - Synonym variants: replace key terms with alternatives
   - Recent preprints: add `2024 OR 2025 OR 2026` to queries
2. **Sources to check:**
   - WebSearch for Google Scholar / general academic results
   - WebFetch on arXiv for preprint-level work
   - WebFetch on Semantic Scholar API for citation-level data
3. **Verdict per gap:**

| Finding | Verdict | Action |
|---------|---------|--------|
| No counterevidence found | `[VERIFIED]` gap is open | Keep as-is |
| Partial counterevidence (adjacent domain, single paper) | `[NARROWED]` gap exists but narrower than claimed | Reframe the gap precisely |
| Direct counterevidence (paper does exactly this) | `[CLOSED]` gap is invalid | Drop from candidate list, cite the paper |

4. **Parallelize:** Launch up to 3 verification agents concurrently to reduce wall-clock time

**Output of this step:** Verified gap list with evidence citations

### Step 4: Approach Synthesis

**Goal:** Select exactly 3 gaps and build them into publication approaches.

**Selection criteria** (pick the best 3 from verified gaps):

| Criterion | Weight | How to Assess |
|-----------|--------|---------------|
| Gap verified & specific | 30% | Must be `[VERIFIED]` or `[NARROWED]`, not `[UNVERIFIED]` or `[CLOSED]` |
| Technical feasibility | 25% | Can this be done with standard tools/compute? Does it match researcher context? |
| Publication venue fit | 20% | Does this fit a Q1/Q2 journal's scope? |
| Novelty magnitude | 15% | Incremental improvement vs. new formulation vs. new paradigm |
| Building-block potential | 10% | Does this enable follow-on work? |

**For each of the 3 selected approaches, construct:**

```markdown
## Approach N: <Title>

### Research Gap (Verified)
<Precise gap statement with citations to existing work and what they don't do>
<Verification status: [VERIFIED] / [NARROWED] / [UNVERIFIED]>

### Proposed Contribution
<What exactly you would do that is new>
<1-2 sentence "elevator pitch" for the paper>

### Method Outline
- System model / problem formulation
- Proposed algorithm or framework
- Key technical differentiator from prior work
- Evaluation approach (baselines, metrics, datasets/simulators)

### Target Venue
<Specific journal(s) with reasoning for venue fit>

### 5-Gates Critical Review

**Gate 1 — First Principles:**
What is the actual problem? Is this the right tool? Are there simpler alternatives?

**Gate 2 — Opportunity Cost:**
What does pursuing this displace? Is this the highest-leverage use of effort?

**Gate 3 — Inversion (What if it fails?):**
| Failure Mode | Probability | Mitigation |
|---|---|---|
| ... | ... | ... |

**Gate 4 — Proportionality:**
Does the solution match the problem scale? Is this a paper-sized contribution or over/under-scoped?

**Gate 5 — Second-Order Effects:**
What happens after success? Career positioning, follow-on research, maintenance burden?

### Evaluation Summary
| Criterion | Score (1-10) | Notes |
|-----------|-------------|-------|
| Scientific novelty | | |
| Publication feasibility | | |
| Technical risk | | |
| Venue fit | | |
| Career value | | |

### Verdict: [RECOMMENDED / VIABLE / RISKY]
```

### Step 5: Comparison & Report Generation

**Goal:** Write the final structured report.

1. Build a **head-to-head comparison table**:

```markdown
| | Approach 1 | Approach 2 | Approach 3 |
|---|---|---|---|
| Gap status | [VERIFIED] | [NARROWED] | [VERIFIED] |
| Novelty | 8/10 | 7/10 | 9/10 |
| Feasibility | 9/10 | 7/10 | 6/10 |
| Risk | LOW | MEDIUM | HIGH |
| Venue | IEEE IoT-J | IEEE TWC | IEEE JSAC |
| Verdict | RECOMMENDED | VIABLE | RISKY |
```

2. Write a **recommendation paragraph** explaining which approach to pursue first and why
3. Save to `docs/publication-scout/YYYY-MM-DD-<topic-slug>-approaches.md`
4. Print console summary

## 5-Gates Critical Thinking Protocol (Reference)

The 5-Gates protocol is applied to each approach. The gates are:

1. **First Principles** — Strip assumptions. What is the actual problem? Is this the right tool? Question whether simpler alternatives exist.
2. **Opportunity Cost** — Every approach displaces something. Is this the highest-leverage option? What do you give up?
3. **Inversion** — Assume the effort fails. What went wrong? Surface risks and failure modes with probability estimates and mitigations.
4. **Proportionality** — Match solution to problem scale. Is this a paper-sized contribution? Not under-scoped (trivial) or over-scoped (thesis-sized)?
5. **Second-Order Effects** — What happens after publication? Career positioning, follow-on research, created dependencies?

## Integration with Other Skills

This skill can invoke or reference:

| Skill | When Used | How |
|-------|-----------|-----|
| `kb-query` | Step 1 | Grep local KB for existing papers and open problems at keyword intersections |
| `paper-discoverer` | Step 1 | Search Semantic Scholar and arXiv APIs for recent papers |
| `research-lookup` | Step 3 | Use Perplexity Sonar for deep research verification (if API key available) |

If these skills are not available, the pipeline falls back to direct WebSearch + WebFetch + Semantic Scholar API calls.

## Error Handling

- **API failures:** If Semantic Scholar or arXiv is unreachable, log and continue with other sources. Mark affected intersections as `[PARTIAL DATA]`.
- **No KB available:** Skip local KB queries (Step 1.2). The skill works without a local KB — it just has less context.
- **Fewer than 3 verified gaps:** If verification closes too many gaps, present whatever verified gaps remain and note: "Only N approaches could be verified. Consider broadening keywords or relaxing constraints."
- **All gaps closed:** Report honestly: "Deep research found that all identified gaps have been recently addressed. The field at this intersection is well-covered. Consider: (a) narrowing to a more specific sub-problem, (b) adding a new keyword dimension, (c) pivoting to a different intersection." List the papers that closed each gap.

## Rate Limiting

- Semantic Scholar: max 100 requests per 5 minutes
- arXiv: max 1 request per 3 seconds
- WebSearch: standard rate limits
- Verification agents: max 3 concurrent

## Example Invocation

```
User: /publication-scout UAV, digital twin, MARL, 6G, VLM

Output:
Step 1: Generating 10 keyword combinations...
Step 1: Landscape mapped. 4 low-saturation intersections found.
Step 2: 7 candidate gaps identified.
Step 3: I've identified 7 candidate gaps. Deep research verification will launch
        parallel agents to check each gap against current literature (2024-2026).
        This takes 3-5 minutes but prevents false novelty claims.
        Verify gaps now? [Y/n]
User: Y
Step 3: Verifying... [3 agents launched]
Step 3: Results: 4 VERIFIED, 2 NARROWED, 1 CLOSED
Step 4: Selecting top 3 approaches...
Step 5: Report written to docs/publication-scout/2026-03-15-uav-dt-marl-6g-vlm-approaches.md

┌─────────────┬──────────────────────────┬───────────┬──────────┬─────────┬────────────┐
│             │ Title                    │ Gap       │ Novelty  │ Risk    │ Verdict    │
├─────────────┼──────────────────────────┼───────────┼──────────┼─────────┼────────────┤
│ Approach 1  │ Decentralized MARL for   │ VERIFIED  │ 8/10     │ LOW-MED │ RECOMMENDED│
│             │ DT sync in multi-UAV     │           │          │         │            │
├─────────────┼──────────────────────────┼───────────┼──────────┼─────────┼────────────┤
│ Approach 2  │ VLM+DT cognitive layer   │ NARROWED  │ 9/10     │ MED-HI  │ VIABLE     │
│             │ for UAV network mgmt     │           │          │         │            │
├─────────────┼──────────────────────────┼───────────┼──────────┼─────────┼────────────┤
│ Approach 3  │ IsaacLab+UavNetSim       │ VERIFIED  │ 7/10     │ LOW     │ RECOMMENDED│
│             │ unified DT platform      │           │          │         │            │
└─────────────┴──────────────────────────┴───────────┴──────────┴─────────┴────────────┘

Recommendation: Start with Approach 1 (lowest risk, highest feasibility).
                Approach 2 is the moonshot. Approach 3 is enabling infrastructure.
```

## Output Format

The report file follows this structure:

```markdown
# Publication Scout Report — <Keywords>

> **Date:** YYYY-MM-DD
> **Keywords:** kw1, kw2, kw3, ...
> **Verification:** [COMPLETE / PARTIAL / SKIPPED]

## Landscape Summary
<Brief overview of saturation at each keyword intersection>

## Approach 1: <Title> — [RECOMMENDED / VIABLE / RISKY]
### Research Gap [VERIFIED / NARROWED / UNVERIFIED]
### Proposed Contribution
### Method Outline
### Target Venue
### 5-Gates Critical Review
### Evaluation Summary

## Approach 2: <Title> — [RECOMMENDED / VIABLE / RISKY]
<same structure>

## Approach 3: <Title> — [RECOMMENDED / VIABLE / RISKY]
<same structure>

## Comparison Table
<head-to-head table>

## Recommendation
<which to pursue first and why>

## Appendix: Verification Evidence
<for each verified/narrowed/closed gap, list the search queries and findings>
```
