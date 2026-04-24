---
name: test-check
description: Audit and improve pytest tests for a stEVE source file. Checks whether existing tests are reasonable (coverage, correct split, style) and adds missing tests. Use when the user asks to check, review, add, or fix tests for a file. Triggers on "check tests", "review tests", "write tests", "add tests", "/test-check", or when a file needs test coverage reviewed or improved.
---

# Test Check for a stEVE File

Audit and improve tests for `$ARGUMENTS` (a Python source file path), following the project's unit / integration test split.

## Steps

1. Read the source file at `$ARGUMENTS`.
2. Identify every public class, method, function, and property.
3. **Audit existing tests** (see section below) — find gaps and issues before writing anything new.
4. Fix existing issues. **Remove**: tests that assert nothing, tests that duplicate another test exactly, tests for private implementation details that are already covered transitively, and tests for code that no longer exists in the source file.
5. For each untested item, decide whether it requires SOFA (integration test) or not (unit test) based on whether it directly or transitively imports `Sofa`.
6. Determine the correct output path:
   - Unit tests → `tests/unit/test_<original_filename>.py`
   - Integration tests → `tests/integration/test_<original_filename>.py`
   - If both kinds are needed, create/extend both files.
7. Check whether the output file(s) already exist. If they do, read them and extend rather than overwrite — add only tests not already present.
8. Write the test file(s) following the rules below.
9. Report: issues found, tests fixed, tests added, and any items intentionally left untested (with reason).

---

## Audit existing tests

Before writing new tests, check for these issues in existing test files:

### Coverage gaps
- Public methods, properties, or functions with no test → flag as missing.
- `StEveObject` subclasses missing a config round-trip test → always required.
- Edge cases absent: `None` defaults, empty sequences, boundary values, expected exceptions.

### Wrong test placement
- SOFA-dependent code tested in `tests/unit/` → move or add `pytest.importorskip("Sofa")`.
- Non-SOFA code in `tests/integration/` unnecessarily → move to `tests/unit/`.

### Missing markers
- Integration tests without `pytestmark = pytest.mark.integration` → add it.
- Integration tests without `Sofa = pytest.importorskip("Sofa")` → add it.

### Poor test quality
- Tests that assert nothing (no `assert` statement) → fix or remove.
- Tests that catch exceptions silently without asserting on them → fix with `pytest.raises`.
- Tests that compare numpy arrays with `==` → replace with `numpy.testing.assert_array_equal` or `assert_allclose`.
- Tests with logic beyond arrange → act → assert (loops, conditionals) → simplify.
- Tests that depend on each other (shared mutable state) → isolate with fixtures.

### Missed Hypothesis opportunities
- Pure, fast serialization or geometry functions tested only with hand-picked values → add property-based tests.

### Style issues
- Test functions not named `test_<what_is_verified>` → rename.
- Class-level tests not grouped in `class Test<ClassName>` → restructure.
- Mutable default arguments or bare `except` clauses → fix.

---

## Test writing rules

### Structure
- One test function per behaviour, named `test_<what_is_verified>`.
- Group tests for the same class in a `class Test<ClassName>` block (no `__init__`, use fixtures instead).
- Keep each test short: arrange → act → assert, with no logic beyond that.

### Unit tests
- No SOFA import; skip the whole module with `Sofa = pytest.importorskip("Sofa")` if any transitive dependency needs it.
- Use `pytest.fixture` for shared setup; prefer function-scoped fixtures.
- For every `StEveObject` subclass, always include a round-trip test:
  ```python
  def test_config_roundtrip(self):
      obj = MyClass(param_a=1, param_b="x")
      restored = MyClass.from_config(**obj.to_config())
      assert restored.param_a == obj.param_a
      assert restored.param_b == obj.param_b
  ```
- Test edge cases: `None` defaults, empty sequences, boundary values.
- Use `pytest.raises` for expected exceptions.
- Prefer `numpy.testing.assert_array_equal` / `assert_allclose` over `==` for numpy arrays.

### Property-based tests (Hypothesis)
For pure, fast, deterministic functions — especially `StEveObject` serialization and geometry utilities — prefer Hypothesis over hand-picked examples:

```python
from hypothesis import given, strategies as st

@given(friction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
       name=st.text(min_size=1))
def test_config_roundtrip_hypothesis(friction, name):
    obj = SomeStEveObject(friction=friction, name=name)
    restored = SomeStEveObject.from_config(**obj.to_config())
    assert restored.friction == obj.friction
    assert restored.name == name
```

Use Hypothesis when:
- The function is pure and fast (no SOFA, no I/O)
- The input space is large or has non-obvious edge cases (floats, strings, nested structures)
- You are testing a serialization round-trip or mathematical invariant

Do **not** use Hypothesis for SOFA-dependent code — simulation steps are too slow and stateful for property-based exploration.

### Integration tests
- Mark every test and/or the whole module:
  ```python
  import pytest
  pytestmark = pytest.mark.integration

  Sofa = pytest.importorskip("Sofa")
  ```
- Wrap `examples/` scripts or minimal end-to-end flows (reset + a few steps).
- Assert on shapes, value ranges, and absence of exceptions — not exact values.

### Style
- Follow the Google Python Style Guide (same as the rest of the project).
- Docstrings on test classes; omit them on individual test functions unless the intent is non-obvious from the name.
- No mutable default arguments; no bare `except`.
- Import only what is needed; keep imports at the top of the file.

---

## Checklist before finishing

- [ ] Existing tests audited for coverage gaps, placement, markers, and quality.
- [ ] All `StEveObject` subclasses have a config round-trip test.
- [ ] Integration tests marked and guarded with `importorskip`.
- [ ] Numpy comparisons use `assert_array_equal` / `assert_allclose`.
- [ ] Hypothesis used where appropriate for pure functions.
- [ ] Report lists: issues fixed, tests added, items intentionally untested.
