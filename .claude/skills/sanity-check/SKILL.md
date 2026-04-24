---
name: sanity-check
description: >
  Add defensive sanity checks (ValueError raises) for invalid input data in a Python file.
  Use when the user says "add sanity checks", "add validation", "add input checks",
  "validate inputs", "/sanity-check", or asks to guard a file against bad data.
  Auto-triggers when reviewing dataclasses or __post_init__ methods that handle
  arrays, coordinates, radii, or other structured numeric data.
---

# Sanity Check

Add defensive `ValueError` raises to `$ARGUMENTS` for invalid input data.

## Steps

1. Read the target file at `$ARGUMENTS`.
2. **Audit existing checks** — before adding anything, scan for checks that should be removed:
   - Checks on values guaranteed valid by the class itself (e.g. array built from `self.coordinates.shape`).
   - Duplicate checks that fire for the same condition as an earlier check in the same method.
   - Checks on internal method arguments (not system-boundary inputs).
   - `assert` statements used as validation → replace with `ValueError` or remove if internal.
   Remove or fix these before proceeding.
3. Identify every class, `__init__`, `__post_init__`, or constructor-like method that
   accepts data from outside (arrays, scalars, sequences, strings, enums).
4. For each parameter, decide which checks apply (see catalogue below).
5. Insert the checks at the top of the relevant method, after any type coercions
   (e.g., `np.array(...)` conversions) but before any use of the value.
6. Re-read the file and verify no duplicate or contradictory checks were added.
7. Report: checks removed, classes/methods modified, checks added.

---

## Check catalogue

### numpy arrays

| Property to verify | Check pattern |
|--------------------|---------------|
| Expected ndim | `if arr.ndim != N:` |
| Expected last axis | `if arr.shape[-1] != 3:` |
| Full shape | `if arr.shape != (N, M):` |
| Minimum length | `if arr.shape[0] < 2:` |
| Length matches another array | `if a.shape[0] != b.shape[0]:` |
| All finite | `if not np.all(np.isfinite(arr)):` |
| Non-negative | `if not np.all(arr >= 0):` |
| Positive | `if not np.all(arr > 0):` |
| In range [lo, hi] | `if not np.all((arr >= lo) & (arr <= hi)):` |

### Scalars (float / int)

| Condition | Check pattern |
|-----------|---------------|
| Positive | `if value <= 0:` |
| Non-negative | `if value < 0:` |
| In range | `if not (lo <= value <= hi):` |
| Finite | `if not math.isfinite(value):` |

### Sequences / containers

| Condition | Check pattern |
|-----------|---------------|
| Non-empty | `if len(seq) == 0:` |
| Minimum length | `if len(seq) < N:` |

### Strings

| Condition | Check pattern |
|-----------|---------------|
| Non-empty | `if not name:` |

---

## Error message format

Every `ValueError` message must identify:
1. The class or context (e.g. `f"Branch '{self.name}': "` or `f"BranchingPoint: "`)
2. What was expected (shape, range, constraint)
3. What was received

```python
# Good
raise ValueError(
    f"Branch '{self.name}': coordinates must have shape (N, 3), "
    f"got {self.coordinates.shape}"
)

# Good — scalar
raise ValueError(
    f"BranchingPoint: radius must be positive, got {self.radius}"
)

# Bad — no context, no actual value
raise ValueError("Invalid coordinates")
```

---

## Placement rules

- Insert checks **after** type coercions (e.g., after `object.__setattr__(self, "x", np.array(x))`
  for frozen dataclasses), so the check sees the canonical type.
- Insert checks **before** any read of the attribute (e.g., before `.flags.writeable = False`).
- For frozen dataclasses that use `object.__setattr__`, the coercion already happened in
  `__post_init__`; add checks immediately after the matching coercion block.
- Do **not** add checks inside property getters or normal methods — validate at construction time.

---

## What NOT to check

- Internal consistency that is guaranteed by the class itself (e.g., an array built
  from `self.coordinates.shape` — it will always have the right shape).
- Return values of numpy operations that are known to be valid by construction.
- Anything already guarded by Python's type system or a library invariant.
- Do not add `assert` statements — use `ValueError` so checks survive `python -O`.

---

## Constraints

- Do NOT change logic, algorithms, or method signatures.
- Do NOT add comments explaining what checks do — the `ValueError` message is self-documenting.
- Follow Google Python Style Guide (absolute imports, no relative imports).
- If `numpy` is not yet imported and a check needs it, add `import numpy as np` at the top.
- If `math` is not yet imported and a scalar finiteness check needs it, add `import math`.
- Run pylint after editing to ensure no new lint issues:
  ```bash
  ~/.pyenv/versions/sofa/bin/python -m pylint "$ARGUMENTS"
  ```
  Fix any issues introduced before reporting done.
