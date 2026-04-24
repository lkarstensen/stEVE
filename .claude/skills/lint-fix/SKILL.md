---
name: lint-fix
description: Run both pylint and pyright on a Python file and fix all reported issues. Use when the user says "fix pylint", "run pylint on", "pylint errors", "lint this file", "fix pyright", "fix type errors", "pylance errors", "type check", "full lint", "fix all lint", "lint and type check", "/lint-fix", "/pylint-fix", "/pyright-fix", or wants both style and type errors fixed in one pass.
---

# Lint Fix

Run pylint **and** pyright on `$ARGUMENTS` and fix every reported issue.

## Steps

1. Read the target file at `$ARGUMENTS`.
2. Run both tools in parallel:
   ```bash
   ~/.pyenv/versions/sofa/bin/python -m pylint "$ARGUMENTS"
   ~/.pyenv/versions/sofa/bin/pyright "$ARGUMENTS"
   ```
3. Fix pylint issues first (style, imports, docstrings).
4. Fix pyright issues (type errors, missing annotations).
5. Re-run both tools to verify clean. If issues remain, fix and re-run (max 3 passes each).
6. Report: issues fixed per tool, issues skipped (with reason).

---

## Pylint: fixable vs unfixable

### Fix automatically

| Symbol | Fix |
|--------|-----|
| `missing-module-docstring` | Add Google-style module docstring (one-line summary). |
| `missing-class-docstring` | Add Google-style class docstring with Args/Attributes sections. |
| `missing-function-docstring` | Add Google-style method docstring. Only add Args/Returns when non-obvious. |
| `line-too-long` | Wrap to ≤100 chars. Break at operators, after commas, using `(` continuation. |
| `trailing-whitespace` | Remove trailing spaces. |
| `missing-final-newline` | Add trailing newline. |
| `wrong-import-order` | Reorder: stdlib → third-party → local. Separate groups with blank line. |
| `wrong-import-position` | Move imports to top of file (after module docstring). |
| `unused-import` | Remove if truly unused; keep if it's a re-export (comment `# noqa` if needed). |
| `ungrouped-imports` | Group stdlib / third-party / local with blank lines between groups. |
| `import-outside-toplevel` | Move import to top unless inside a function for circular-import reason — if so, add `# pylint: disable=import-outside-toplevel` with a comment explaining why. |
| `bad-whitespace` / `bad-continuation` | Fix spacing to match PEP 8. |
| `unnecessary-pass` | Remove `pass` if the block has other statements. Keep if it's the only statement. |
| `superfluous-parens` | Remove unnecessary parentheses. |
| `use-implicit-booleaness-not-comparison` | Replace `if x == []` with `if not x`, etc. |
| `singleton-comparison` | Replace `if x == None` with `if x is None`. |
| `redefined-builtin` | Rename parameter (e.g. `list` → `items`, `type` → `kind`). |
| `consider-using-f-string` | Convert `%`/`.format()` string to f-string. **Exception**: skip if the string is the first argument of any log call (`_logger.debug/info/warning/error/critical`) — logging must use lazy `%` formatting (W1203). |
| `no-else-return` | Remove `else` after `return`. |
| `unnecessary-comprehension` | Simplify `[x for x in y]` to `list(y)`. |

### Report but do not auto-fix

| Symbol | Why skip |
|--------|----------|
| `too-many-arguments` | May need refactor to dataclass or kwargs — flag to user. |
| `too-many-locals` | Same — flag to user. |
| `too-many-branches` | Refactoring decision. |
| `duplicate-code` | Needs architectural judgement. |
| `cyclic-import` | Fix requires module restructure. |
| `abstract-method` | Intentional in some base classes. |
| `protected-access` | May be intentional in tests. |
| Any `E` (error) not in fix list | Flag immediately — these may indicate real bugs. |

---

## Pylint: docstring rules (Google Style)

**Module docstring** — one-line summary ending in period:
```python
"""A one-line summary of the module, terminated by a period."""
```

**Class docstring** — summary + Args + Attributes. Do NOT duplicate type annotations:
```python
"""One-line summary.

Args:
    param_a: What it controls.

Attributes:
    state: Current state, populated by ``reset()``.
"""
```

**`@abstractmethod`** — always has a docstring defining the contract. Concrete overrides omit docstring when behaviour matches the abstract base's contract.

**Method docstring** — one-liner. Add Args/Returns/Raises only when non-obvious from signature:
```python
"""Advance the environment by one timestep."""
```

**Property** — single line, no Returns section needed.

Never write multi-paragraph or multi-line comment blocks. No task/fix references in docstrings.

---

## Pylint: import ordering (Google Style)

```python
"""Module docstring."""

# 1. __future__
from __future__ import annotations

# 2. stdlib
import os
import time
from typing import List

# 3. third-party
import numpy as np
import gymnasium as gym

# 4. local (absolute only — never relative)
from steve.core.steveobject import StEveObject
from steve.util.logging import get_logger
```

Blank line between each group. Alphabetical within group.

---

## Pylint: line wrapping

Prefer continuation inside `()`:
```python
# before
some_long_variable_name = some_function(argument_one, argument_two, argument_three, argument_four)

# after
some_long_variable_name = some_function(
    argument_one, argument_two, argument_three, argument_four
)
```

For imports:
```python
from some.long.module import (
    ClassOne,
    ClassTwo,
    function_three,
)
```

---

## Pyright: fixable vs unfixable

### Fix automatically

| Category | Fix |
|----------|-----|
| `reportArgumentType` — `None` passed where `T` expected | Narrow with `if x is not None` guard, or use `assert x is not None` if provably safe. |
| `reportReturnType` — missing or wrong return type | Add or correct the return annotation; add explicit `return` if missing. |
| `reportAttributeAccessIssue` — attribute does not exist | Fix typo, add the attribute, or add a type: ignore with explanation. |
| `reportOperatorIssue` — operator not supported on type | Cast or guard with `isinstance`. |
| `reportIndexIssue` — subscript on non-subscriptable type | Add generic annotation or cast. |
| `reportAssignmentType` — wrong type assigned | Correct the annotation or the expression. |
| `reportOptionalMemberAccess` — attribute access on `X \| None` | Add `if x is not None` guard before access. |
| `reportOptionalSubscript` — subscript on `X \| None` | Same: guard before subscript. |
| `reportOptionalCall` — call on `X \| None` | Guard before call. |
| `reportPossiblyUnbound` — variable may be unbound | Initialize variable before conditional assignment. |
| `reportMissingTypeArgument` — generic used without type args | Add type arguments (e.g. `list` → `list[str]`). |
| `reportUnknownVariableType` / `reportUnknownParameterType` | Add explicit type annotation. |
| `reportUnnecessaryIsInstance` — isinstance always true/false | Remove dead branch or fix type. |
| `reportUnnecessaryComparison` — comparison always true/false | Remove dead branch or fix type. |
| Missing annotation on public function parameter | Add annotation matching the evident type from usage context. |

### Report but do not auto-fix

| Category | Why skip |
|----------|----------|
| `reportMissingModuleSource` / `reportMissingImports` | SOFA stubs not present — expected in this project. Add `# type: ignore[import]` only if user confirms. |
| `reportUnknownMemberType` on third-party libs | Stubs incomplete — not fixable without stubs. |
| `reportPrivateUsage` | May be intentional in tests. |
| Any error inside generated or vendored code | Do not modify vendored code. |
| Errors that require logic changes to fix correctly | Flag to user with explanation. |

---

## Pyright: fix strategy

**Prefer narrowing over `# type: ignore`.** Only use `# type: ignore` when:
- The third-party library has no stubs and the code is correct.
- The error is a pyright false-positive with no clean fix.

When using `# type: ignore`, always append the error code:
```python
result = some_untyped_lib.call()  # type: ignore[no-untyped-call]
```

**Do NOT add `from __future__ import annotations`** unless it already exists — it changes runtime semantics for dataclasses and Pydantic models.

---

## Pyright: common patterns in this codebase

```python
# Guard for Optional
if self.radii is not None and other.radii is not None:
    return np.array_equal(self.radii, other.radii)

# Assert when provably safe (caller already checked)
assert x is not None  # proven by prior guard in caller

# Annotate numpy arrays
coords: np.ndarray  # shape (N, 3)
```

---

## Conflict resolution

When a fix for one tool would break the other (rare), prefer the fix that eliminates both issues. If impossible, fix the error-level issue and note the conflict.

---

## Constraints

- Use `~/.pyenv/versions/sofa/bin/python -m pylint` and `~/.pyenv/versions/sofa/bin/pyright` — never bare tool names.
- Do NOT change logic — only style, formatting, docstrings, types, and annotations.
- Do NOT add `# pylint: disable=` comments to silence warnings unless genuinely unfixable and intentional. If using a disable, always add a one-line comment explaining why.
- Preserve existing inline comments.
- Do not suppress `reportMissingModuleSource` for SOFA — it is expected and already configured.
- Target: pylint 10.00/10 and pyright 0 errors before reporting done.
