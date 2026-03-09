---
name: kb-maintenance
description: >
  Run a full KB update cycle: discover new papers, filter, extract, integrate, and report. Orchestrates the other four KB skills.
layer: meta
domain: [general]
source-project: papers-kb
depends-on: [paper-extractor, kb-integrator, paper-discoverer, kb-query]
tags: [knowledge-base, orchestration, pipeline, automation, discovery]
---

# KB Maintenance — Full Update Cycle Orchestrator

## Purpose

Run a complete knowledge base update cycle by orchestrating the four component skills:
1. `paper-discoverer` — find new papers
2. `paper-extractor` — extract paper cards
3. `kb-integrator` — classify and integrate
4. `kb-query` — health check queries

## When to Use

Invoke this skill when:
- Running a scheduled daily update (via launchd or manual trigger)
- The user asks to "update the KB" or "check for new papers"
- New PDFs have been added to `pdf/` and need processing

**Not for:** Individual paper extraction or one-off queries.

## Full Cycle Steps

### Step 1: Pre-Flight Check

1. Read `kb/registry.json` — get current paper count and existing IDs
2. Read `kb/stats.md` — get last update date
3. Check `pdf/` for any new PDFs not in the registry (compare filenames)
4. Check `pdf/downloads/` for PDFs the user has moved to `pdf/` since last cycle
5. Report: "Starting cycle. KB has N papers. Found M new PDFs to process."

### Step 2: Discovery (5-10 min)

Invoke the `paper-discoverer` skill:
- Run all standing queries from `kb/config/search-queries.md`
- Download available PDFs for auto-accepted candidates
- Write candidate report

**Skip this step** if invoked with `--skip-discovery` or if the user only wants to process existing PDFs.

### Step 3: Process New PDFs

Identify PDFs to process:
- PDFs in `pdf/` that are not in `kb/registry.json` (match by filename)
- PDFs recently moved from `pdf/downloads/` to `pdf/`

For each new PDF, invoke `paper-extractor`:
- **Parallelize** using the Agent tool: launch one agent per PDF (up to 5 concurrent)
- Each agent runs the paper-extractor skill independently
- Wait for all agents to complete

**Agent prompt template**:
```
You are a paper extraction agent. Read the PDF at <path> and create a paper card following the paper-extractor skill format.

Read the taxonomy at kb/config/taxonomy.md for domain tags.

Save the card to kb/papers/<generated-name>.md

Follow the paper card format exactly as specified in the paper-extractor skill.
```

### Step 4: Integration (Sequential)

For each new paper card in `kb/papers/` with `novelty: pending`:

Invoke `kb-integrator` **sequentially** (one at a time):
1. Classify novelty
2. Update topic files
3. Update index, timeline, registry

**Order**: Process papers chronologically (oldest first) so SoTA classifications are stable.

### Step 5: Health Check

Run these `kb-query` checks:
1. Registry integrity: `total_papers` matches actual entries count
2. Index completeness: every registry entry has an index row
3. Topic coverage: every domain tag in taxonomy has at least one topic file
4. Orphan check: no paper cards without registry entries

### Step 6: Write Cycle Report

Write `kb/reports/YYYY-MM-DD-cycle.md`:

```markdown
# KB Update Cycle — YYYY-MM-DD

## Summary

- Cycle started: HH:MM
- Cycle completed: HH:MM
- Papers before: N
- Papers after: M
- New papers added: M-N
- Topics updated: K
- Candidates discovered: L
- PDFs downloaded: P

## Papers Added

| Paper | Novelty | Topics Updated |
|-------|---------|---------------|
| <id> | sota | drone-simulation |
| <id> | incremental | rl-for-flight |

## Discovery Results

- Queries executed: N
- Auto-accepted: M
- Pending review: K
- Discarded: L

## Health Check

- [ ] Registry count matches: YES/NO
- [ ] Index complete: YES/NO
- [ ] Topic coverage: YES/NO
- [ ] No orphans: YES/NO

## Errors

<Any errors encountered during the cycle, or "None">
```

### Step 7: Git Commit (if changes)

If any KB files changed:

```bash
git add kb/
git commit -m "kb: daily update YYYY-MM-DD — N papers added, M topics updated"
```

Do NOT commit PDFs in `pdf/downloads/` (large binaries).

## Error Recovery

### Failed extraction
- Log the error in the cycle report
- Skip the paper and continue with the next
- Do NOT update the registry for failed extractions

### Failed integration
- Log the error
- The paper card exists but registry is not updated
- Next cycle will retry (card has `novelty: pending`)

### API failures during discovery
- Log which APIs failed
- Continue with available sources
- Note in cycle report for manual retry

## Running Modes

### Full cycle (default)
All steps: discovery → extraction → integration → health check → report

### Process-only (--skip-discovery)
Skip step 2. Only process new PDFs already in `pdf/`.

### Discovery-only (--discovery-only)
Only run step 2 (discovery). No extraction or integration.

### Health-check-only (--health-check)
Only run step 5 (health check) and report.

## Automated Execution

For daily automated runs via launchd, use:

```bash
claude --print --dangerously-skip-permissions "Run the kb-maintenance skill with a full cycle"
```

The launchd plist at `~/Library/LaunchAgents/com.papers.kb-update.plist` handles scheduling.
