---
name: book-reader
description: >
  Reads an entire textbook, extracts mathematical/technical content, and builds a structured knowledge base plus actionable agent skills.
layer: meta
domain: [general]
source-project: Book-Mathematical-Foundation-of-RL
depends-on: []
tags: [knowledge-extraction, textbook, methodology]
---

# Book Reader Agent — Comprehensive Textbook Knowledge Extraction

## Purpose
Read an entire textbook (PDF or other format), extract all mathematical/technical content, and build a structured knowledge base + actionable agent skills from it.

## When to Use
When the user provides a textbook or technical book and wants to:
- Build a persistent knowledge base from it
- Create agent skills that encode the book's reasoning frameworks
- Transform static content into an actionable reasoning toolkit

---

## Phase 0: Environment Preparation

### 0.1 Check PDF Reading Capability
```bash
# Test if poppler-utils is available (needed by Read tool for PDFs)
which pdftoppm 2>/dev/null && echo "OK" || echo "MISSING"
```

If missing, install a Python PDF reader FIRST before any reading:
```bash
# Create a persistent venv for PDF reading
uv venv /tmp/pdfreader && source /tmp/pdfreader/bin/activate && uv pip install pymupdf
```

Then use this helper to read PDFs:
```bash
source /tmp/pdfreader/bin/activate && python3 -c "
import fitz
doc = fitz.open('PATH_TO_PDF')
for i in range(len(doc)):
    print(f'=== PAGE {i+1} ===')
    print(doc[i].get_text())
"
```

**CRITICAL:** Verify PDF reading works on a small file BEFORE launching parallel agents. If sub-agents cannot read PDFs, they will waste time and fail.

### 0.2 Inventory the Book
```bash
# List all PDF files to understand book structure
find /path/to/book -name "*.pdf" | sort
```

Identify: chapters, appendices, supplementary material, code examples.

### 0.3 Create Output Directories
```bash
mkdir -p /home/ai/.claude/projects/<project-path>/memory/
mkdir -p /path/to/repo/.claude/skills/
```

---

## Phase 1: Reading Strategy

### 1.1 First Pass — Table of Contents + Structure
Read the table of contents and preface/overview FIRST. This gives:
- Chapter dependencies (which chapters build on which)
- Key concepts per chapter
- The book's pedagogical arc
- Total scope (number of chapters, pages)

### 1.2 Chapter Reading — Parallel Batch Strategy

**For books with 8+ chapters:** Read in parallel batches of 3-5 chapters using background Task agents. Group by topic dependency:

```
Batch 1: Foundational chapters (definitions, basic concepts)
Batch 2: Core methods chapters
Batch 3: Advanced methods chapters
Batch 4: Appendices + supplementary material
```

**For each chapter, the reading agent prompt should specify:**

```
Read [PDF path] and extract ALL of the following:

1. DEFINITIONS: Every formal definition with its number and exact statement
2. THEOREMS/LEMMAS: Complete statements with ALL conditions and conclusions
3. PROOFS: Key proof steps and techniques (not line-by-line, but the strategy)
4. ALGORITHMS: Full pseudocode with input/output/steps
5. KEY EQUATIONS: With equation numbers, in LaTeX-style notation
6. EXAMPLES: Illustrative examples that clarify concepts
7. RELATIONSHIPS: How concepts in this chapter connect to other chapters
8. REMARKS: Author's insights, common mistakes, practical advice

Use LaTeX-style notation for all math. Be extremely precise.
Return content organized by section within the chapter.
```

### 1.3 Handling Large Chapters (>20 pages)
Split into page ranges: pages 1-20, then 21-40, etc. Merge results.

### 1.4 Handling Failed Reads
If an agent fails (PDF tool issues, timeout), read that chapter directly from the main conversation using the Bash+pymupdf approach. Do NOT re-launch a failing agent pattern.

---

## Phase 2: Knowledge Base Construction

### 2.1 File Organization Principles

Create **topic-oriented** KB files, NOT chapter-by-chapter files:
- Group related content across chapters into coherent topics
- Each KB file should be self-contained for its topic
- Include cross-references to other KB files
- Target 100-250 lines per file (enough detail, not overwhelming)

### 2.2 KB File Structure Template

```markdown
# [Topic Name]

**Sources:** [Chapter numbers]

## 1. [First Major Concept]

### Definition
[Precise mathematical definition]

### Key Equations
[Numbered equations in code blocks]

### Theorems
[Full statements with conditions]

### Algorithms
[Pseudocode]

## 2. [Second Major Concept]
...

## N. Cross-References
[Links to related KB files and concepts]
```

### 2.3 What to Include vs Exclude

**Include:**
- Every theorem statement (with ALL conditions — incomplete conditions are dangerous)
- Every algorithm's pseudocode
- Key equations with their equation numbers from the book
- Convergence conditions and guarantees
- Relationships between concepts
- Practical implications and failure modes

**Exclude:**
- Lengthy proof details (include proof *strategy* only)
- Numerical examples (unless they illustrate a non-obvious point)
- Historical context / motivation paragraphs
- Exercises (unless they contain important results)

### 2.4 Recommended KB File Count

| Book Size | KB Files | Lines Each |
|-----------|----------|------------|
| < 100 pages | 3-4 | 100-150 |
| 100-300 pages | 6-10 | 150-250 |
| 300-600 pages | 10-15 | 150-250 |
| 600+ pages | 15-20 | 150-250 |

Always create one **cross-cutting patterns/taxonomy** file that synthesizes across all chapters.

---

## Phase 3: Skill Construction

### 3.1 Skill Design Principles

Skills should encode **reasoning procedures**, not just knowledge:
- Given X, how to derive Y
- Given a problem, which theorem/technique to apply
- Step-by-step analysis procedures
- Decision trees for classification

### 3.2 Standard Skill Set for Technical Books

1. **Theory Analyzer:** Classify and analyze instances of the book's subject matter
2. **Designer/Builder:** Compose the book's building blocks to create new things
3. **Prover/Verifier:** Verify correctness using the book's theoretical framework
4. **Implementer:** Translate the book's ideas into working code
5. **Innovator:** Meta-skill that orchestrates the others for novel exploration

### 3.3 Skill File Structure Template

```markdown
# [Skill Name]

## Purpose
[1-2 sentences]

## When to Use
[Bullet list of trigger conditions]

## Procedure

### Step 1: [Action]
[Detailed instructions with decision points]

### Step 2: [Action]
...

## Reference Tables
[Quick-lookup tables mapping inputs to techniques/theorems]

## Output Format
[What the skill should produce when invoked]
```

### 3.4 Skill Content Guidelines
- Include concrete decision trees (if X then use theorem Y)
- Include template equations that can be filled in
- Reference specific KB file sections for detailed content
- Include "anti-patterns" (common mistakes to avoid)
- Keep skills actionable, not encyclopedic

---

## Phase 4: Memory Index

Create MEMORY.md (max 200 lines) as the entry point:

```markdown
# [Book Title] Knowledge Base

## Book Reference
[Author, title, publisher, year, location on disk]

## Knowledge Base Files
[Table: filename | topics | source chapters]

## Agent Skills
[Table: skill name | purpose]

## Key Results Quick Reference
[5-10 most important theorems/results, one line each]

## Concept Progression
[How the book's ideas build on each other, as ASCII diagram]
```

---

## Phase 5: Verification

### 5.1 Completeness Check
For each chapter, verify:
- [ ] All theorems extracted (compare count with table of contents)
- [ ] All algorithms extracted
- [ ] Key equations present with correct numbers

### 5.2 Cross-Reference Check
- [ ] Every KB file is referenced in MEMORY.md
- [ ] Skills reference appropriate KB files
- [ ] No orphan files

### 5.3 Functional Test
Pick a known result from the book and test the skills:
- Can the theory-analyzer correctly classify it?
- Can the convergence-prover verify its properties?
- Can the implementer produce working code for it?

---

## Lessons Learned / Common Pitfalls

1. **PDF reading fails silently in sub-agents:** Always verify PDF reading capability BEFORE launching parallel agents. Install pymupdf proactively.

2. **Sub-agents have independent environments:** They cannot use packages installed in the main session's venv. Either install system-wide or ensure they create their own venv.

3. **Large PDF text extraction can exceed output limits:** Use page ranges (20 pages at a time) and save to files for large chapters.

4. **Mathematical notation gets mangled in PDF extraction:** Always cross-check extracted equations against the original PDF. Common issues: subscripts/superscripts lost, summation symbols garbled, matrix notation flattened.

5. **Don't create KB files chapter-by-chapter:** Topic-oriented files are far more useful. A concept like "convergence" spans multiple chapters and should be in one place.

6. **Skills should be procedures, not encyclopedias:** A skill that says "here are 50 theorems" is less useful than one that says "given problem type X, apply theorem Y by checking conditions Z."

7. **MEMORY.md must stay under 200 lines:** It's loaded into every conversation. Keep it as an index, not a knowledge dump.

8. **Parallel agent launching is key for speed:** Reading 10 chapters sequentially takes 10x longer. Launch 4-5 reading agents in parallel, then create all KB files in parallel.

9. **Always read the book yourself for the KB files:** Agent extractions are raw material. The KB files should be curated, organized, and cross-referenced — not just copied agent output.

10. **Test with a simple query after building:** Ask "prove that Q-learning converges" or equivalent for your domain. If the KB + skills can't handle it, iterate.
