---
name: book-to-knowledge-base
description: >
  Losslessly converts technical books from PDF chapters into structured markdown knowledge bases, preserving every definition, theorem, algorithm, equation, and proof with LaTeX notation and chapter-dependency metadata.
layer: meta
domain: [general]
source-project: Book-Mathematical-Foundation-of-RL
depends-on: []
tags: [knowledge-extraction, pdf-to-markdown, textbook, lossless, knowledge-base]
---

# Book to Knowledge Base — Lossless PDF-to-Markdown Conversion

## Purpose

Convert a technical book (PDF chapters) into a structured markdown knowledge base that preserves **all** mathematical and technical content. The output is a self-contained directory of markdown files — one per chapter, plus an overview and cross-reference index — suitable for AI-assisted retrieval via `Read`/`Grep`.

## When to Use

Invoke this skill when:
- You have a technical/mathematical book as one or more PDF files
- You want a **lossless** markdown knowledge base (not a summary or curated extraction)
- The goal is a persistent reference that can be queried by AI agents
- You need chapter-dependency tracking for structured navigation

**Not for:** Quick summaries, topic-oriented reorganization, or non-technical books. For curated/topic-oriented KB construction, see the `book-reader` skill.

## Key Design Decisions

**Chapter-by-chapter, not topic-oriented.** Each output file maps 1:1 to a book chapter. This preserves the author's pedagogical structure, makes verification trivial (compare file to PDF), and avoids lossy reorganization. Cross-cutting queries are handled by the cross-reference index.

**Lossless, not curated.** Include everything: every definition, theorem (with ALL conditions), algorithm, equation, proof strategy, example, and remark. Judgment calls about what to exclude introduce errors. Let the reader/agent decide what's relevant at query time.

**Small enough for direct access.** Target total KB size under 50K tokens (~8,000 lines). This fits comfortably in `Read`/`Grep` access without needing a vector DB or MCP server.

---

## Output Specification

### Directory Structure

```
knowledge_base/
  00-overview.md                    # Reading order, dependency diagram, chapter summary table
  01-<chapter-slug>.md              # Chapter 1
  02-<chapter-slug>.md              # Chapter 2
  ...
  NN-<chapter-slug>.md              # Last chapter
  (NN+1)-appendix.md               # Appendix (if book has one)
  (NN+2)-cross-reference-index.md  # Master index across all chapters
```

### File Naming

- Two-digit zero-padded prefix: `01-`, `02-`, ..., `12-`
- Slug derived from chapter title: lowercase, hyphens for spaces
- Examples: `01-basic-concepts.md`, `07-temporal-difference-methods.md`, `11-appendix.md`

### Chapter File Format

Every chapter file follows this exact structure:

```markdown
---
chapter: <number or "appendix">
title: <Chapter Title>
key_topics: [topic1, topic2, topic3, ...]
depends_on: [1, 2, 3]
required_by: [5, 6, 7]
---

# Chapter N: <Title>

> Source: *<Book Title>* (<Author>, <Publisher> <Year>), Chapter N, pp. X-Y
> Supplemented by: <lecture slides, errata, etc. — or omit if none>
> Errata: <corrections applied — or "No corrections for this chapter">

## Purpose and Context

<2-4 paragraphs: what this chapter covers, why it matters, position in the
book's arc, relationship to preceding/following chapters>

---

## N.1 <First Section Title>

<Content following the book's section structure exactly>

---

## N.2 <Second Section Title>

...

---

## Summary and Key Takeaways

<Bulleted list of the chapter's main results>
```

### YAML Frontmatter Fields

| Field | Type | Description |
|-------|------|-------------|
| `chapter` | int or string | Chapter number, or `"appendix"`, or `"overview"` |
| `title` | string | Chapter title from the book |
| `key_topics` | string[] | All major topics covered, for search/filtering |
| `depends_on` | int[] | Chapter numbers that are prerequisites |
| `required_by` | int[] | Chapter numbers that depend on this one |

### Overview File Format (00-overview.md)

```markdown
---
type: overview
title: Knowledge Base Overview
---

# Knowledge Base Overview

<1-2 paragraphs: book title, author, scope, part structure>

---

## Reading Order and Dependency Diagram

<ASCII art showing chapter dependencies>

---

## Chapter Summary Table

| Chapter | Title | Key Topics | Pages | Algorithms Covered |
|---------|-------|------------|-------|--------------------|
| 1 | ... | ... | 1-13 | (none) |
| 2 | ... | ... | 15-34 | ... |

---

## Concept Flow Diagram

<ASCII art showing how concepts progress through the book>

---

## File Listing

| File | Description |
|------|-------------|
| 00-overview.md | This file. |
| 01-... | Chapter 1: ... |

---

## Navigation Tips for AI Agents

<5-8 numbered tips for how an AI agent should query this knowledge base>
```

### Cross-Reference Index Format

```markdown
---
type: synthesis
title: Cross-Reference Index and Algorithm Comparison
---

# Cross-Reference Index and Algorithm Comparison

## 1. Master Concept Index
| Concept | Defined In | Also Used In |
...

## 2. Algorithm Comparison Table
| Algorithm | Type | Model Required? | ... |
...

## 3. Theorem and Lemma Index
| Theorem | Name | One-Line Statement | Chapter | Key Conditions |
...

## 4. Equation Quick-Reference
<Top ~20 equations grouped by topic, in LaTeX>

## 5. Key Transitions
<Major conceptual shifts across the book, as tables>
```

---

## Pipeline

### Phase 0: Environment & Inventory

**Step 0.1 — Verify PDF reading capability**

```bash
# Option A: Claude Code Read tool (works if poppler-utils installed)
# Try reading a small PDF first

# Option B: pymupdf fallback
uv run --with pymupdf python3 -c "
import fitz
doc = fitz.open('PATH_TO_PDF')
for i in range(min(3, len(doc))):
    print(f'=== PAGE {i+1} ===')
    print(doc[i].get_text())
"
```

**CRITICAL:** Verify PDF reading produces clean text on 2-3 pages BEFORE proceeding. If math symbols are garbled, try a different extraction method. Do not proceed with broken extraction.

**Step 0.2 — Inventory all files**

List all PDFs and identify: chapter PDFs, combined PDF, table of contents, appendices, errata, supplementary material (slides, code).

**Step 0.3 — Create output directory**

```bash
mkdir -p /path/to/repo/knowledge_base
```

### Phase 1: Table of Contents & Structure

Read the table of contents, preface, and overview to extract:

1. **Chapter list** with titles and page ranges
2. **Chapter dependencies** (which chapters build on which)
3. **Book structure** (parts, sections per chapter)
4. **Supplementary material** mapping (which slides/code maps to which chapter)
5. **Errata** (corrections to apply during extraction)

Save as a working document. This informs everything that follows.

### Phase 2: Chapter-by-Chapter Extraction

Process each chapter using the extraction protocol below. For books with 8+ chapters, use the parallelization strategy (see below).

#### Extraction Protocol (per chapter)

Read the entire chapter (20 pages at a time for large chapters). Extract ALL of the following:

| Category | What to Extract | Formatting |
|----------|----------------|------------|
| **Definitions** | Every formal definition with its number and exact statement | Block with "**Definition N.M**:" prefix |
| **Theorems/Lemmas** | Complete statements with ALL conditions and conclusions | Block with "**Theorem N.M**:" prefix; list conditions explicitly |
| **Proof strategies** | Key technique and steps (not line-by-line) | Under theorem, as "**Proof sketch**:" |
| **Algorithms** | Full pseudocode with input/output/steps | As "**Algorithm N.M: Name**" with numbered steps |
| **Equations** | Every numbered equation, in LaTeX | `$$...\tag{N.M}$$` format |
| **Examples** | All worked examples that illustrate concepts | Under the relevant section |
| **Remarks** | Author's insights, practical advice, common mistakes | As "**Remark**:" or "**Important note**:" |
| **Relationships** | How this chapter connects to other chapters | In "Purpose and Context" section and inline forward/back references |

#### Formatting Rules

1. **LaTeX notation** for all math: `$inline$` and `$$display$$`
2. **Section numbering** mirrors the book: `## N.1`, `## N.2`, etc.
3. **Equation tags**: Use `\tag{N.M}` matching the book's equation numbers
4. **Tables** for structured data (state transition tables, reward tables, comparison tables)
5. **Horizontal rules** (`---`) between major sections
6. **Block quotes** for source citations: `> Source: *Book Title* ...`
7. **Bold** for definitions, theorem names, algorithm names
8. **Forward/back references** as inline text: "See Chapter N" or "as shown in equation (M.K)"
9. **Errata corrections**: Apply silently, note in the source citation block

#### Per-Chapter Quality Gate

Before moving to the next chapter, verify:
- [ ] Every section from the book has a corresponding `## N.M` heading
- [ ] All theorems have complete conditions (no missing assumptions)
- [ ] All algorithms have complete pseudocode
- [ ] Equation numbering matches the book
- [ ] YAML frontmatter `depends_on` and `required_by` are accurate

### Phase 3: Synthesis Files

After all chapters are extracted:

**Step 3.1 — Overview file (00-overview.md)**

Build from the Phase 1 structure document. Include:
- ASCII dependency diagram (drawn from the `depends_on`/`required_by` metadata)
- Chapter summary table (one row per chapter with: title, key topics, page range, algorithms)
- Concept flow diagram showing the book's intellectual progression
- File listing with one-line descriptions
- Navigation tips for AI agents

**Step 3.2 — Cross-reference index**

Scan ALL chapter files to build:
1. **Master concept index**: Every major concept → where defined, where used
2. **Algorithm comparison table**: All algorithms in one table with type, model requirement, convergence guarantee, on/off-policy
3. **Theorem/lemma index**: All theorems with one-line statements, chapter, key conditions
4. **Equation quick-reference**: The ~20 most important equations grouped by topic
5. **Key transitions**: Major conceptual shifts across the book (e.g., model-based → model-free, tabular → function approximation)

### Phase 4: Verification

**Step 4.1 — Completeness audit**

For each chapter, compare the markdown file against the PDF:
- Count of definitions matches
- Count of theorems/lemmas matches
- Count of algorithms matches
- All numbered equations present
- No sections skipped

**Step 4.2 — Cross-reference integrity**

- Every concept in the cross-reference index points to a real section
- Every `depends_on` / `required_by` reference is reciprocal (if Ch 3 depends on Ch 2, then Ch 2's `required_by` includes 3)
- File listing in overview matches actual files

**Step 4.3 — Sample query test**

Pick 3 known results from the book and verify they can be found via:
1. Concept lookup in `cross-reference-index.md`
2. Direct `Grep` across `knowledge_base/*.md`
3. Reading the specific chapter file

### Phase 5: Integration

**Step 5.1 — Update project CLAUDE.md**

Add a "Knowledge Base" section documenting:
- File map (table of files → content)
- How to query the KB (concept lookup, keyword search, chapter dependencies)
- Design decision: no MCP server (KB is small enough for direct Read/Grep)

**Step 5.2 — Report**

Output a summary:
- Number of files created with line counts
- Total line count and estimated token count
- Any chapters or sections with extraction issues
- Sample query demonstrating the KB works

---

## Parallelization Strategy

### For books with 8+ chapters

Read chapters in parallel using background agents. Group by dependency tiers:

```
Tier 1 (no dependencies):     Ch 1, Appendix
Tier 2 (depends on Tier 1):   Ch 2, Ch 3
Tier 3 (depends on Tier 2):   Ch 4, Ch 5, Ch 6
Tier 4 (depends on Tier 3):   Ch 7, Ch 8, Ch 9, Ch 10
```

Launch 3-5 agents per tier. Each agent receives:
1. The chapter PDF (or page range of the combined PDF)
2. The extraction protocol (copy the full protocol from this skill)
3. The output path
4. The errata relevant to this chapter

**Agent prompt template:**

```
Read [PDF_PATH] pages [START]-[END] and convert Chapter [N]: "[TITLE]" to markdown.

Follow this exact format:

[Paste the Chapter File Format section from this skill]

Extraction rules:
[Paste the Extraction Protocol section from this skill]

Apply these errata corrections: [LIST]

Save output to: [OUTPUT_PATH]
```

After all agents complete, verify in the main conversation and build the synthesis files.

### For single-PDF books

Use the Read tool with page ranges: `pages: "1-20"`, `pages: "21-40"`, etc. Process sequentially or launch agents with page ranges.

---

## Quality Standards

### Mathematical Precision

- **Theorem conditions**: NEVER omit conditions. A theorem without its conditions is wrong. If the theorem states "for all $\gamma \in [0, 1)$", include that.
- **Notation consistency**: Use the book's notation throughout. Don't switch between $v(s)$ and $V(s)$ unless the book does.
- **LaTeX accuracy**: Verify subscripts, superscripts, summation bounds, and matrix notation. PDF extraction commonly mangles these.

### Structural Fidelity

- **Section mapping**: Every section heading `N.M` in the book gets a `## N.M` heading in the markdown
- **No reorganization**: Don't move content between sections. If the book puts Example 3.2 in Section 3.4, keep it there.
- **Preserve sequence**: Definitions → theorems → proofs → examples → remarks, in the order they appear

### Completeness Targets

| Book Size | Expected Output | Lines per Chapter |
|-----------|----------------|-------------------|
| < 150 pages | 3,000-5,000 lines total | 200-400 |
| 150-300 pages | 5,000-8,000 lines total | 400-700 |
| 300-500 pages | 8,000-12,000 lines total | 500-800 |
| 500+ pages | 12,000+ lines total | 500-800 |

---

## Common Failure Modes

### 1. PDF text extraction garbles math

**Symptom:** Subscripts lost, summation symbols become "P", matrices flatten to single lines.
**Fix:** Cross-check extracted text against the PDF visually. For heavily formatted math, read the PDF page as an image and transcribe manually. Consider using lecture slides as a secondary source — they often have cleaner text extraction.

### 2. Sub-agent can't read PDFs

**Symptom:** Agent reports "file not found" or empty extraction.
**Fix:** Verify PDF reading works BEFORE launching agents (Phase 0). If the Read tool fails, pre-extract to text files:
```bash
uv run --with pymupdf python3 -c "
import fitz
doc = fitz.open('book.pdf')
for i in range(len(doc)):
    with open(f'/tmp/page-{i+1:03d}.txt', 'w') as f:
        f.write(doc[i].get_text())
"
```
Then give agents the text files instead of the PDF.

### 3. Chapter exceeds agent context window

**Symptom:** Agent output truncates mid-chapter, missing later sections.
**Fix:** Split into 20-page chunks. Give each chunk to a separate agent, then merge results. Alternatively, process the chapter in the main conversation where context is managed automatically.

### 4. Theorem conditions are incomplete

**Symptom:** Theorem states a result but the conditions under which it holds are vague or missing.
**Fix:** This is the most dangerous failure mode. Cross-check every theorem against the PDF. If conditions are unclear, check the errata and lecture slides. Mark uncertain conditions with `[VERIFY]`.

### 5. Equation numbering drifts

**Symptom:** Equation (7.5) in the markdown is actually (7.6) in the book.
**Fix:** After completing a chapter, do a sequential pass comparing equation tags against the PDF. This is fastest as a visual scan.

### 6. Cross-reference index is stale

**Symptom:** Index references sections that were renamed or reorganized during editing.
**Fix:** Build the cross-reference index LAST, after all chapter files are finalized. Generate it by scanning the actual files, not from memory.

---

## Worked Example: Reference Output

The knowledge base at `Book-Mathematical-Foundation-of-Reinforcement-Learning/knowledge_base/` is the reference implementation of this skill. It converts a 10-chapter, 270-page RL textbook into:

- 13 files, ~7,900 lines total (~45K tokens)
- Chapter files range from 355 lines (Ch 1, foundational definitions) to 838 lines (Ch 8, most algorithms)
- Overview file: 180 lines with dependency diagram, chapter summary table, concept flow, navigation tips
- Cross-reference index: 323 lines with concept index (80+ entries), algorithm comparison (15 algorithms), theorem index (20+ theorems), equation quick-reference (~20 equations)

Use this as a template for structure, formatting, and level of detail.
