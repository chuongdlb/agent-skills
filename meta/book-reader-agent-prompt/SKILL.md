---
name: book-reader-agent-prompt
description: >
  Prompt template for the /agent command to launch a book-reading agent that builds knowledge bases and skills from textbooks.
layer: meta
domain: [general]
source-project: Book-Mathematical-Foundation-of-RL
depends-on: [book-reader]
tags: [prompt-template, agent, methodology]
---

# Book Reader Agent — Prompt Template for /agent

## Usage

Copy the instruction below, replace the `[PLACEHOLDERS]`, and paste into `/agent`.

---

## Instruction (copy from here)

```
Read the entire book at [PATH_TO_PDF] and build a structured knowledge base + agent skills from it.

## STEP 0: ENVIRONMENT SETUP

1. Check if the Read tool can handle PDFs:
   - Try reading the PDF with the Read tool first (it works if poppler-utils is installed)
   - If it fails, install pymupdf:
     ```
     uv venv /tmp/pdfreader && source /tmp/pdfreader/bin/activate && uv pip install pymupdf
     ```
   - Then use this pattern to read pages:
     ```
     source /tmp/pdfreader/bin/activate && python3 -c "
     import fitz
     doc = fitz.open('[PATH_TO_PDF]')
     for i in range(START, END):
         print(f'=== PAGE {i+1} ===')
         print(doc[i].get_text())
     "
     ```
2. VERIFY reading works on pages 1-3 BEFORE doing anything else. If it fails, stop and report.

## STEP 1: READ TABLE OF CONTENTS

Read the first 10-15 pages to extract:
- Full table of contents with page numbers
- Chapter dependencies (which build on which)
- Total chapter count and page ranges

Save this to: [MEMORY_DIR]/book-structure.md

## STEP 2: READ ALL CHAPTERS

Read every chapter, 20 pages at a time. For each chapter extract ALL of:
1. DEFINITIONS: Every formal definition with its number and exact statement
2. THEOREMS/LEMMAS: Complete statements with ALL conditions and conclusions
3. ALGORITHMS: Full pseudocode with input/output/steps
4. KEY EQUATIONS: With equation numbers, using LaTeX-style notation
5. PROOF STRATEGIES: Not line-by-line, but the key technique used
6. EXAMPLES: Only those that clarify non-obvious points
7. RELATIONSHIPS: How concepts connect to other chapters
8. REMARKS: Author's insights, common mistakes, practical advice

Use LaTeX-style notation for all math. Be extremely precise with conditions on theorems.

Save raw extractions to temp files: /tmp/chapter-[N]-raw.md

## STEP 3: BUILD KNOWLEDGE BASE

Create TOPIC-ORIENTED (not chapter-by-chapter) knowledge base files at:
[MEMORY_DIR]/

Rules:
- Group related content from MULTIPLE chapters into coherent topics
- Each file should be 100-250 lines, self-contained for its topic
- Include cross-references between KB files
- Target [6-15] files depending on book size (6 for <200 pages, 10 for 200-400, 15 for 400+)
- Always create one cross-cutting "design patterns" or "taxonomy" file

Each KB file structure:
```markdown
# [Topic Name]
**Sources:** [Chapter numbers]

## 1. [Concept]
### Definition
### Key Equations
### Theorems (with ALL conditions)
### Algorithms

## N. Cross-References
```

What to INCLUDE: Every theorem statement (complete conditions), every algorithm, key equations with numbers, convergence conditions, relationships between concepts.
What to EXCLUDE: Lengthy proof details (strategy only), routine numerical examples, historical motivation, exercises.

## STEP 4: BUILD AGENT SKILLS

Create 3-5 skill files at: [SKILLS_DIR]/

Standard skill set:
1. **[domain]-theory-analyzer.md** — Classify and analyze instances of the book's subject matter. Include decision trees mapping problem types to applicable theorems/techniques.
2. **[domain]-designer.md** — Compose the book's building blocks to create new things. Include the design patterns library and a step-by-step design procedure.
3. **[domain]-convergence-prover.md** (if mathematical) — Verify correctness using the book's theoretical framework. Include proof templates.
4. **[domain]-implementer.md** — Translate ideas into working code. Include code templates and a debugging guide.
5. **[domain]-innovator.md** — Meta-skill that orchestrates the others for novel exploration.

Each skill structure:
```markdown
# [Skill Name]
## Purpose
## When to Use
## Procedure (step-by-step with decision points)
## Reference Tables (input -> technique/theorem mappings)
## Anti-Patterns (common mistakes)
## Output Format
```

Skills must encode REASONING PROCEDURES, not just knowledge. "Given X, check Y, apply Z" is good. "Here are 50 theorems" is bad.

## STEP 5: UPDATE MEMORY INDEX

Create/update [MEMORY_DIR]/MEMORY.md (MAX 200 lines):
- Book reference (author, title, path)
- Table of all KB files with topics and source chapters
- Table of all skills with purposes
- 5-10 most important results (one line each)
- Concept progression diagram (ASCII)

## STEP 6: VERIFY

For each chapter, confirm:
- All theorems extracted (compare with table of contents)
- All algorithms extracted
- Key equations present with correct numbers

List any gaps found.

## OUTPUT

When done, report:
- Number of KB files created with line counts
- Number of skill files created
- Any chapters or sections that had extraction issues
- A sample query demonstrating the KB works (e.g., look up a known theorem)
```

---

## Placeholder Reference

| Placeholder | Example | Description |
|-------------|---------|-------------|
| `[PATH_TO_PDF]` | `/home/ai/books/rl-textbook.pdf` | Absolute path to the PDF |
| `[MEMORY_DIR]` | `/home/ai/.claude/projects/-home-ai-source-myproject/memory` | Persistent memory directory |
| `[SKILLS_DIR]` | `/home/ai/source/myproject/.claude/skills` | Skills directory in the repo |
| `[domain]` | `rl`, `stats`, `optim` | Short domain prefix for skill filenames |
| `[6-15]` | `10` | Target KB file count based on book size |

## Tips

- For books over 300 pages, the agent may run out of context. Split into two runs: "Read chapters 1-5 and build KB files for them" then "Read chapters 6-10 and extend the KB".
- If the agent struggles with PDF reading, pre-convert to text: `source /tmp/pdfreader/bin/activate && python3 -c "import fitz; doc=fitz.open('book.pdf'); [open(f'/tmp/ch{i}.txt','w').write(doc[i].get_text()) for i in range(len(doc))]"`
- The agent works best when you specify the domain explicitly (e.g., "This is a reinforcement learning textbook" vs just "read this book").
