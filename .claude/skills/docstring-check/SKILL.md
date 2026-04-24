---
name: docstring-check
description: Audit and improve Google-style docstrings and inline comments in a stEVE Python file. Checks whether existing docstrings and comments are appropriate and complete, not just adds new ones. Use when the user asks to check, review, add, or fix docstrings or comments in a file. Triggers on "check docstrings", "review docstrings", "add docstrings", "fix docstrings", "check comments", "/docstring-check", or when a file needs documentation reviewed or improved.
---

# Docstring Check for a stEVE File

Audit and improve docstrings and inline comments in `$ARGUMENTS` (a Python file path) following the project's documentation style.

## Steps

1. Read the file at `$ARGUMENTS`.
2. **Audit existing docstrings** (see section below) — identify issues before writing anything new.
3. **Audit existing inline comments** (see section below) — identify superfluous and missing comments.
4. Fix all issues found. **Remove**: redundant type-annotation prose, `__init__` docstrings when `Args:` is in the class docstring, overriding method docstrings identical to base, `Returns:` sections on properties, WHAT-comments and commented-out dead code. Fix spacing, stale content, and format violations.
5. For every module, class, method, function, and property that is missing a docstring, write one following the rules below.
6. Do not change any logic or signatures.
7. Write the updated file back and report: docstring issues fixed, docstrings added, comment issues fixed, and anything intentionally left without a docstring (with reason).

---

## Audit existing docstrings

Before writing new docstrings, check for these issues:

### Wrong or missing sections
- Class docstring missing `Args:` for `__init__` parameters → add it (args go in class docstring, not `__init__`).
- Class docstring missing `Attributes:` for public instance attributes set after construction (e.g. by `reset()`) → add it.
- Method docstring missing `Args:` / `Returns:` / `Raises:` where behaviour is non-obvious from signature → add missing sections.
- `Args:` / `Returns:` / `Raises:` present when behaviour IS obvious from signature → remove the noise.

### Redundant content
- Type annotations repeated in prose in docstring sections → remove (type hints in signature are authoritative).
- Overriding methods with identical docstring as base class → remove the override's docstring entirely.
- `__init__` methods with their own docstring when `Args:` is already in the class docstring → remove.

### Format violations
- Summary line exceeds 80 characters → shorten.
- Summary line does not end with `.`, `?`, or `!` → fix punctuation.
- Single-quotes used (`'''`) instead of double-quotes (`"""`) → fix.
- Property docstring has a `Returns:` section → remove it (properties use single-line only).
- `Args:` section lists `self` → remove it.

### Wrong placement
- `Args:` block in `__init__` instead of class docstring → move to class level.

### Stale content
- Docstring refers to parameters, attributes, or behaviour that no longer exist in the code → update or remove.
- Docstring describes old class name or module path → update.

---

## Audit existing inline comments

### Remove these
- Comments that restate what the code already says (`i += 1  # increment i`) → delete.
- Comments that describe WHAT, not WHY (`# call reset` above `self.reset()`) → delete.
- Comments referencing the task, fix, or caller (`# added for the Y flow`, `# used by X`) → delete.
- Commented-out dead code → delete unless there is a clear stated reason to keep it.

### Fix these
- Less than two spaces before `#` → fix spacing (`code  # comment`).
- Comments that were accurate but the code changed → update or remove.
- Heavy `# ---...---` banner dividers → replace with light `# — section name —` style.
- In large files (many methods, multiple logical groups), add `# — section name —` dividers if absent.

### Add these (sparingly)
- Hidden constraints not obvious from the code (e.g. SOFA requires a specific call order).
- Subtle invariants that, if violated, would cause silent failures.
- Workarounds for specific external bugs (name the bug/ticket if possible).
- Behaviour that would genuinely surprise a reader unfamiliar with the domain.

**Default is no comment.** Only add when removing the comment would confuse a future reader.

---

## Docstring rules

This project follows the [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html).

- **Module**: one-line summary terminated by a period at the top of the file. Add a usage example block if the module has a non-obvious entry point.
- **Class**: summary sentence(s), then `Args:` for every `__init__` parameter (placed in the *class* docstring, not `__init__`), then `Attributes:` for any public instance attributes set after construction (e.g. by `reset()`). Do not repeat type annotations in prose.
- **`@abstractmethod`**: always has a docstring — it defines the contract implementors must fulfill. Include `Raises: NotImplementedError` only if the base provides no default and callers need to know.
- **Method / function**: one-line summary. Add `Args:`, `Returns:`, and/or `Raises:` sections only when the behaviour is non-obvious from the signature. Skip the docstring entirely on concrete overrides whose behaviour does not materially differ from the abstract base's documented contract.
- **Property**: single-line docstring only, no `Returns:` section.
- **Inline comments**: at least two spaces before `#`; only when the *why* is non-obvious.
- Summary lines must not exceed 80 characters and must end with `.`, `?`, or `!`.
- Use `triple-double-quotes` (`"""`) for all docstrings.
- Do **not** document `self` in `Args:`.

---

## Checklist before finishing

- [ ] All existing docstrings audited for wrong sections, redundant content, format violations, wrong placement, and staleness.
- [ ] `Args:` in class docstring (not `__init__`).
- [ ] No type annotations repeated in prose.
- [ ] No docstrings on overriding methods that don't differ from base.
- [ ] Properties use single-line docstring only.
- [ ] Summary lines ≤ 80 chars, end with `.`/`?`/`!`.
- [ ] All docstrings use `"""`.
- [ ] Inline comments audited: WHAT-comments removed, spacing fixed, stale comments updated, WHY-comments added only where genuinely non-obvious.
- [ ] No commented-out dead code remaining (unless explicitly justified).
- [ ] Report lists: docstring issues fixed, docstrings added, comment issues fixed, items intentionally left undocumented.
