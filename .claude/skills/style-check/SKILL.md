---
name: style-check
description: Google Style Guide checker for Python files. Reviews and fixes compliance with the Google Python Style Guide. Use when the user says "style check", "check style", "google style", "style guide", "/style-check", or asks to review a file for style compliance.
---

# Google Style Guide checker

Review and fix `$ARGUMENTS` (a Python file path) for compliance with the
[Google Python Style Guide](http://google.github.io/styleguide/pyguide.html).

## Steps

1. Read the file at the path given in `$ARGUMENTS`.
2. Check every rule in the checklist below.
3. **Auto-fix all violations** directly in the file using the Edit tool. Do not ask for permission — just fix them. The only exception is changes that would alter observable runtime behaviour (see "Report only" below).
4. For violations that would alter observable behaviour, report them with line numbers and a concrete suggestion, but do not apply them automatically.
5. After all fixes are applied, print a short summary:
   - How many violations were auto-fixed
   - How many observable-behaviour changes were flagged (with details)
   - Overall pass/fail verdict

---

## Auto-fix vs Report-only

**Auto-fix** — apply without asking:
- Import ordering (stdlib → third-party → local, lexicographic within groups)
- `Optional[x]` → `x | None`, `typing.Tuple` → `tuple`, `typing.List` → `list`, etc.
- Implicit optional `a: str = None` → `a: str | None = None`
- Trailing semicolons
- Whitespace inside `()`, `[]`, `{}`
- Whitespace before `,`, `;`, `:`
- Logging f-string → `%`-style equivalent (e.g. `f"x={x}"` → `"x=%s", x`)
- `'''` docstrings → `"""`
- Missing space around binary operators
- Extra/missing blank lines between top-level definitions (PEP 8 two-blank-line rule)
- Splitting or restructuring long lines (>100 chars)
- Renaming identifiers to fix naming convention violations (snake_case, CapWords, etc.)
- Changing mutable default arguments to `None` + initialise inside body
- Replacing `assert` with `raise ValueError` in production code (non-test files)
- Adding or rewriting docstrings (Google-style)
- Removing or rewriting non-compliant comprehensions

**Report only** (flag but do not auto-apply):
- Restructuring `try/except` blocks in ways that change exception propagation
- Any fix where multiple valid interpretations exist and the wrong choice would change runtime behaviour

---

## Checklist

### Imports
- [ ] Only `import x` for packages/modules; `from x import y` where x is a package
- [ ] No relative imports — always use full package names
- [ ] Standard abbreviations only for `import y as z` (e.g. `numpy as np`)
- [ ] Import order: `__future__` → stdlib → third-party → local; lexicographic within each group

### Naming
- [ ] Modules: `lower_with_under`
- [ ] Classes: `CapWords`
- [ ] Functions and methods: `lower_with_under()`
- [ ] Constants: `CAPS_WITH_UNDER`
- [ ] Protected/internal names: single leading `_`
- [ ] No single-character names except loop counters, `e` (exception), `f` (file handle)
- [ ] No dashes in package or module names

### Formatting
- [ ] Maximum line length 100 characters (exceptions: long URLs, unavoidable imports)
- [ ] No backslash line continuations — use implicit joining inside `()`, `[]`, `{}`
- [ ] Indentation: 4 spaces, never tabs
- [ ] No whitespace inside `()`, `[]`, `{}`
- [ ] No whitespace before `,`, `;`, `:`; single space after
- [ ] Single space around binary operators
- [ ] When a type annotation is present, spaces around `=` for default parameter values
- [ ] No vertically aligned tokens across consecutive lines
- [ ] No trailing semicolons

### Strings
- [ ] Use `"""` (not `'''`) for multi-line strings and docstrings
- [ ] No `+` / `+=` for string accumulation in loops — use `''.join(list)` instead
- [ ] Logging calls pass a string literal as the first argument, not an f-string

### Exceptions
- [ ] No bare `except:` and no catching `Exception`/`BaseException` without re-raising
- [ ] No `assert` for enforcing preconditions in production code
- [ ] `try` blocks contain minimal code; cleanup in `finally`
- [ ] Prefer built-in exception types (`ValueError`, `TypeError`, etc.) for standard cases

### Type Annotations
- [ ] Public APIs are annotated
- [ ] `self` and `cls` are not annotated unless required
- [ ] `str | None` union syntax used instead of `Optional[str]`
- [ ] No implicit optional (`a: str = None`) — use `a: str | None = None`
- [ ] Abstract types used for parameters (`Sequence` not `list`, `Mapping` not `dict`)
- [ ] Built-in generics used (`tuple[int, str]` not `typing.Tuple[int, str]`)

### Functions and methods
- [ ] Public functions have docstrings
- [ ] Mutable default arguments not used — use `None` and initialise inside the body
- [ ] Lambdas fit on one line (≤ 60–80 chars); longer logic uses a named function
- [ ] `staticmethod` avoided in favour of module-level functions

### Classes
- [ ] Classes have docstrings
- [ ] Constructor args documented in the **class** docstring `Args:` section (not `__init__`)
- [ ] Public instance attributes set outside `__init__` documented in `Attributes:` section
- [ ] `classmethod` used only for named constructors or class-level state mutations

### Docstrings (Google style)
- [ ] Summary line ≤ 80 characters, ends with `.`, `?`, or `!`
- [ ] `Args:` section present when parameters are non-obvious
- [ ] `Returns:` / `Yields:` section present when return value is non-obvious
- [ ] `Raises:` section lists all explicitly raised exceptions
- [ ] Type annotations not repeated in docstring prose
- [ ] `@abstractmethod` methods always have a docstring (they define the contract)
- [ ] Concrete overrides omit docstrings when behaviour matches the abstract base's contract

### Comprehensions
- [ ] No more than one `for` clause and one optional filter per comprehension
- [ ] Comprehensions prefer readability over conciseness

### Main guard
- [ ] Executable scripts put logic in a `main()` function
- [ ] `if __name__ == '__main__':` guard present in executable files
