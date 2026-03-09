---
name: paper-discoverer
description: >
  Search academic databases and GitHub for new papers relevant to the knowledge base, score relevance, and produce a ranked candidate list.
layer: meta
domain: [general]
source-project: papers-kb
depends-on: []
tags: [discovery, academic-search, semantic-scholar, arxiv, github, knowledge-base]
---

# Paper Discoverer — Academic Paper Discovery Pipeline

## Purpose

Search Semantic Scholar, arXiv, and GitHub for new papers relevant to the knowledge base. Score each candidate for relevance, auto-accept high-scoring papers, and produce a ranked candidate list for human review.

## When to Use

Invoke this skill when:
- Running a `kb-maintenance` discovery cycle
- The user asks to find new papers on a topic
- You want to expand the KB's coverage

**Not for:** Extracting papers (use `paper-extractor`), or integrating papers (use `kb-integrator`).

## Input

- `kb/config/search-queries.md` — standing search queries
- `kb/registry.json` — existing papers (for dedup and citation overlap)
- `kb/config/scoring-rubric.md` — relevance scoring criteria

## Output

- `kb/candidates/YYYY-MM-DD-candidates.md` — ranked candidate list
- Downloaded PDFs in `pdf/downloads/` (for auto-accepted papers with available PDFs)
- Updated `kb/candidates/pending-review.md`

## Discovery Pipeline

### Step 1: Load Context

1. Read `kb/config/search-queries.md` for queries
2. Read `kb/registry.json` for existing paper IDs, DOIs, arXiv IDs, and title hashes
3. Read `kb/config/scoring-rubric.md` for scoring criteria

### Step 2: Search Semantic Scholar (Primary)

Use WebFetch to query the Semantic Scholar API:

```
GET https://api.semanticscholar.org/graph/v1/paper/search?query=<query>&limit=20&fields=title,authors,year,venue,externalIds,citationCount,abstract,citations
```

For each query in search-queries.md:
1. Execute the search
2. For each result, check dedup against registry (DOI, arXiv ID, title hash)
3. If not a duplicate, add to candidate list

**Citation expansion**: For each existing KB paper with a Semantic Scholar ID, fetch its citations:
```
GET https://api.semanticscholar.org/graph/v1/paper/<paper_id>/citations?fields=title,authors,year,venue,externalIds,abstract&limit=50
```

### Step 3: Search arXiv

Use WebFetch to query the arXiv API:

```
GET http://export.arxiv.org/api/query?search_query=<query>&start=0&max_results=20&sortBy=submittedDate&sortOrder=descending
```

Parse the Atom XML response. Extract: title, authors, published date, arXiv ID, abstract, categories.

### Step 4: Search GitHub

Use Bash to run `gh search repos`:

```bash
gh search repos --topic=<topic> --sort=updated --limit=10
```

For each repo, check README and recent releases for linked papers (arXiv links, DOI links).

Also check monitored repos from search-queries.md for new paper references:
```bash
gh api repos/<owner>/<repo>/readme --jq '.content' | base64 -d | grep -oE 'arxiv\.org/abs/[0-9]+\.[0-9]+'
```

### Step 5: Score Candidates

For each unique candidate, compute relevance score (0-10):

| Component | Points | How to Assess |
|-----------|--------|---------------|
| Query match | 0-3 | Check title and abstract against KB domain tags |
| Citation overlap | 0-3 | Count shared references with KB papers |
| Recency | 0-2 | Calculate age from publication date |
| Venue quality | 0-2 | Check venue against known top venues list |

### Step 6: Filter and Act

| Score | Action |
|-------|--------|
| >= 5 | Auto-accept: attempt PDF download, add to extraction queue |
| 3-4 | Human review: add to `kb/candidates/pending-review.md` |
| < 3 | Discard: do not include in candidate list |

### Step 7: Download Available PDFs

For auto-accepted papers:

**arXiv papers**: Download from `https://arxiv.org/pdf/<arxiv-id>.pdf`
```bash
curl -L -o "pdf/downloads/<author><year>-<keyword>.pdf" "https://arxiv.org/pdf/<arxiv-id>.pdf"
```

**Open access papers**: Check Semantic Scholar `openAccessPdf` field.

**GitHub-linked papers**: Check repo for PDF links.

Save downloaded PDFs to `pdf/downloads/` — they must be manually moved to `pdf/` before extraction.

### Step 8: Write Candidate Report

Write `kb/candidates/YYYY-MM-DD-candidates.md`:

```markdown
# Discovery Candidates — YYYY-MM-DD

## Auto-Accepted (Score >= 5)

| Title | Authors | Year | Score | Source | PDF |
|-------|---------|------|-------|--------|-----|
| ... | ... | ... | 7 | Semantic Scholar | downloaded |

## Pending Review (Score 3-4)

| Title | Authors | Year | Score | Source | Reason |
|-------|---------|------|-------|--------|--------|
| ... | ... | ... | 4 | arXiv | New method but narrow domain |

## Statistics

- Queries executed: N
- Total candidates found: N
- Duplicates filtered: N
- Auto-accepted: N
- Pending review: N
- Discarded: N
```

### Step 9: Update Pending Review

Append new pending-review entries to `kb/candidates/pending-review.md`:

```markdown
## Pending Review

| Date | Title | Score | Source | Action |
|------|-------|-------|--------|--------|
| YYYY-MM-DD | <title> | 4 | arXiv | [ ] Accept / [ ] Reject |
```

## Rate Limiting

- Semantic Scholar: max 100 requests per 5 minutes (no API key needed)
- arXiv: max 1 request per 3 seconds
- GitHub: standard `gh` CLI rate limits

Add appropriate delays between API calls.

## Error Handling

- If an API is unavailable, log the error and continue with other sources
- If PDF download fails, mark as "unavailable" in the candidate report
- If a query returns 0 results, log it but don't treat as an error

## Top Venues List

For venue scoring (2 points):
- **Robotics**: ICRA, IROS, RSS, CoRL, RA-L, T-RO
- **ML/AI**: NeurIPS, ICML, ICLR, AAAI, IJCAI
- **Systems**: SIGGRAPH, SoCC
- **Vision**: CVPR, ICCV, ECCV
