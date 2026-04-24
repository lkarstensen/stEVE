---
name: full-check
description: Run all quality checks on a stEVE Python file in the correct order: lint, style, sanity, logging, lint again, docstrings, simplify, tests, review. Use when the user asks to "full check", "check everything", "run all checks", or invokes /full-check on a file.
---

# Full Check for a stEVE File

Run all quality checks on `$ARGUMENTS` in dependency order.

## Steps

Run each skill in sequence on the same file. Complete each step fully before moving to the next. **Do not pause for user input between steps — run all 9 steps to completion in a single uninterrupted pass, then output the final report.**

### 1. lint-fix
Fix all pylint and pyright errors first. Clean foundation before any other changes.

### 2. style-check
Check Google Style Guide compliance on the now-clean code.

### 3. sanity-check
Add defensive `ValueError` raises for invalid inputs. This modifies code.

### 4. logging-check
Audit and improve logging. This modifies code.

### 5. lint-fix (second pass)
Re-run pylint and pyright. Steps 3 and 4 may have introduced new violations.

### 6. docstring-check
Audit and improve docstrings on the now-stable, validated, logged code.

### 7. simplify
Review the now-clean, documented code for reuse, quality, and efficiency. Fix any issues found.

### 8. test-check
Audit and improve tests reflecting the final state of the file.

### 9. review
Run the built-in `review` skill on the file's branch changes. Catches any remaining design, correctness, or security issues after all automated fixes.

---

## Final report

After all steps complete, output a single summary with one section per step:

```
## Full Check Report: <filename>

### 1. lint-fix
<what was fixed or "no issues">

### 2. style-check
<what was fixed or "no issues">

### 3. sanity-check
<checks removed / checks added or "no changes">

### 4. logging-check
<calls removed / issues fixed / calls added or "no changes">

### 5. lint-fix (second pass)
<what was fixed or "no issues">

### 6. docstring-check
<content removed / issues fixed / docstrings added or "no changes">

### 7. simplify
<duplications removed / simplifications made or "no changes">

### 8. test-check
<tests removed / tests fixed / tests added or "no changes">

### 9. review
<design / correctness / security issues found or "no issues">
```
