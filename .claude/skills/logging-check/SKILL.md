---
name: logging-check
description: Audit and improve Python logging in a stEVE source file using steve.util.logging.get_logger. Use when the user asks to check, review, add, or fix logging in a file, or instrument a component with log statements. Triggers on "check logging", "review logging", "add logging to", "instrument", "/logging-check", or when a file is being refactored and needs logging reviewed or added.
---

# Logging Check for a stEVE File

Audit and improve logging in `$ARGUMENTS` using the project's `steve.util.logging.get_logger` utility.

## Steps

1. Read the file at `$ARGUMENTS`.
2. **Audit existing logging** (see section below) — identify issues before adding anything.
3. Fix existing issues: correct levels, rewrite f-strings to `%` args, replace `logging.getLogger` with `get_logger`. **Remove** redundant/noisy calls, calls inside tight inner loops, and calls on trivial no-op methods.
4. Check if `from steve.util.logging import get_logger` is already present. If not, add it.
5. Identify missing logging candidates (see section below).
6. Add the import and module-level logger if not present.
7. Add log calls at each missing candidate site.
8. Report every issue found and every change made: which methods got which log calls at which level.

---

## Audit existing logging

Before adding new log calls, check the file for these issues:

### Wrong logger construction
- `logging.getLogger(...)` → replace with `get_logger(__name__)` from `steve.util.logging`.
- `import logging` used without `get_logger` → replace.

### Wrong format style (Pylint W1203)
- f-strings in log calls: `_logger.debug(f"...")` → rewrite as `_logger.debug("...", ...)` with `%` args.
- `.format()` in log calls → same fix.

### Wrong log level
- INFO used for per-step/per-substep values → downgrade to DEBUG.
- DEBUG used for error conditions or unrecoverable states → upgrade to WARNING/ERROR.
- WARNING used for normal operational events → downgrade to INFO or DEBUG.

### Logging inside tight inner loops
- Any log call inside `for _ in range(n_steps)` or similar tight loops → move outside or guard with a counter.

### Duplicate loggers
- Both `_logger` and `self.logger` present without the original `self.logger` being pre-existing → consolidate to `_logger`.

### Large object logging
- Full numpy arrays logged → replace with `.shape` or `.tolist()` for small arrays (≤ 5 elements).

### Redundant / noisy messages
- Messages that restate what the caller already logged in the same call stack → remove.
- Trivial no-op methods (Dummy classes) with log calls → remove unless the call is meaningful.

---

## Import and logger setup

Add after existing imports, before class definitions:

```python
from steve.util.logging import get_logger

_logger = get_logger(__name__)
```

If the class uses `self.logger` already (e.g. `SofaBeamAdapter`), replace `logging.getLogger(...)` with `get_logger(__name__)` and keep using `self.logger` — do not add a second `_logger`.

---

## What to log and at which level

### INFO — always log these
- Component initialized with key config params (in `__init__` or first `reset`)
- Episode reset started (log episode number)
- Target reached / terminal condition triggered
- Simulation (re-)initialized (mesh path, instrument count, dt)
- Error/recovery events (NaN tracking, simulation error flag set)

### DEBUG — log these (gated by STEVE_LOG_LEVEL=DEBUG)
- `reset()` entry and completion with timing (`time.time()`)
- `step()` completion with timing and key scalar outputs (reward, inserted lengths, action)
- Intermediate computed values useful for debugging (insertion point, seed)
- Config params on construction if verbose context helps

### WARNING — log these
- Unexpected state that is handled but not fatal (NaN corrected, clipped action)
- Fallback to default values

### ERROR — log these
- Unrecoverable simulation states before raising exceptions

---

## Timing pattern

For `reset()` and `step()` methods, wrap with timing:

```python
import time  # add to imports if not present

def reset(self, ...):
    _logger.debug("MyClass.reset episode %d", episode_number)
    start_time = time.time()
    # ... existing body ...
    elapsed_ms = (time.time() - start_time) * 1000
    _logger.debug("MyClass.reset completed in %.1fms", elapsed_ms)
```

Only add `import time` if it is not already imported.

---

## Message format rules

- Use lazy `%` formatting — **never f-strings or `.format()`** in logging calls. Pylint enforces this (`logging-fstring-interpolation`).
- Keep messages short: `"Reset episode %d: mesh=%s", n, path` not `"Now resetting episode number %d with mesh path %s", n, path`.
- Log scalar values directly; for arrays, log shape or summary: `"action=%s", action.tolist()` or `"dof shape=%s", arr.shape`.
- Never log full numpy arrays unless they are small (≤ 5 elements).

```python
# correct
_logger.debug("MyClass.reset episode %d", episode_number)
_logger.info("Config saved to %r", file_path)
_logger.warning("NaN corrected at step %d", step)

# wrong — triggers W1203
_logger.debug(f"MyClass.reset episode {episode_number}")
```

---

## What NOT to log

- Inside tight inner loops (e.g., per-SOFA-substep `for _ in range(n_steps)` — log once after the loop).
- Trivial no-op methods (Dummy classes, `step() -> None: ...`).
- Private helper methods unless they contain error-handling logic.
- Anything already logged by a caller in the same call stack.

---

## Checklist before finishing

- [ ] All existing log calls audited for level, format, and placement issues.
- [ ] No `logging.getLogger(...)` or raw `import logging` remaining.
- [ ] Absolute import used: `from steve.util.logging import get_logger` (not relative).
- [ ] `import time` added if timing calls were added.
- [ ] No duplicate loggers (`_logger` and `self.logger` coexist only when `self.logger` was already there).
- [ ] No logging inside per-substep inner loops.
- [ ] Lazy `%` formatting used throughout — no f-strings in log calls.
- [ ] Report lists: issues fixed, new calls added, anything left unchanged and why.
