# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**stEVE** (simulated EndoVascular Environment) is a Python framework for simulating endovascular interventions (minimally invasive medical procedures inside blood vessels). It implements the [Farama Gymnasium](https://gymnasium.farama.org/) interface for use in reinforcement learning research. The physics backend is [SOFA Framework](https://www.sofa-framework.org/) via the BeamAdapter plugin.

## Commands

```bash
# Install in editable mode with dev dependencies
python3 -m pip install -e ".[dev]"

# Verify installation
python3 examples/function_check.py

# Lint (pylintrc is configured)
pylint steve/

# Run all unit tests (no SOFA required)
pytest tests/unit/

# Run integration tests (requires SOFA install)
pytest tests/integration/

# Run a single test file
pytest tests/unit/test_confighandler.py

# Run unit tests only (skip integration even if SOFA is present)
pytest -m "not integration"
```

## Testing

The test suite uses **pytest** with two layers:

- **`tests/unit/`** — No SOFA dependency. Covers `StEveObject` serialization, `ConfigHandler` round-trips, vessel geometry utilities, and `Reward`/`Terminal`/`Truncation`/`Observation` logic. These run anywhere and should be the default CI target.
- **`tests/integration/`** — Requires a SOFA install. Wraps the `examples/` scripts as smoke tests. Mark each test with `@pytest.mark.integration` and use `pytest.importorskip("Sofa")` to skip automatically when SOFA is not available.

Add `pytest` and any test dependencies to a `requirements-dev.txt` (not `setup.py`).

## Logging

The framework uses Python's standard `logging` module, configured via environment variables:

- **`STEVE_LOG_LEVEL`**: Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). Default: `INFO`.
- **`STEVE_LOG_FILE`**: Log file path. If set, logs to both file and console. If unset, console only.

Usage:

```bash
# Log to console at DEBUG level
export STEVE_LOG_LEVEL=DEBUG
python3 examples/function_check.py

# Log to file + console
export STEVE_LOG_LEVEL=DEBUG STEVE_LOG_FILE=steve.log
python3 examples/function_check.py
```

In code, use:

```python
from steve.util.logging import get_logger

logger = get_logger(__name__)
logger.info("Episode started")
logger.debug("Step took %.1fms", elapsed_ms)
```

Logs include:
- Episode lifecycle events (reset, target reached, max steps)
- Step timing and reward/termination state
- Component initialization and state transitions

**When adding logging to new files**, always use `get_logger(__name__)` from
`steve.util.logging` — never `logging.getLogger()` directly. This ensures
env-var configuration applies. Use the absolute import
`from steve.util.logging import get_logger` (Google Style Guide mandates absolute
imports over relative ones).

**Log message format**: use lazy `%` formatting — never f-strings or `.format()` in
log calls. Pylint enforces this via `W1203 logging-fstring-interpolation`.

```python
# correct
_logger.debug("Step took %.1fms", elapsed_ms)
_logger.info("Config saved to %r", file_path)

# wrong
_logger.debug(f"Step took {elapsed_ms:.1f}ms")
```

## Architecture

The framework is built around **declarative composition**: users build an `Env` by wiring together swappable components. All components inherit from `StEveObject` and support YAML serialization.

### Component Hierarchy

```
Env (gymnasium.Env)
└── Intervention          # orchestrates a single episode
    ├── VesselTree        # 3D mesh + centerlines of blood vessels
    ├── Instrument(s)     # catheter/guidewire shape definitions
    ├── Simulation        # SOFA physics engine
    ├── Fluoroscopy       # imaging/tracking feedback (x-ray or 2D/3D coordinates)
    └── Target            # navigation destination (random, branch end, manual…)
├── Observation           # RL observation space
├── Reward                # RL reward signal
├── Terminal              # episode termination condition
├── Truncation            # step limit / timeout
├── Start                 # episode initialization strategy
├── Pathfinder            # optional path planning
├── InterimTarget         # optional waypoint management
├── Info                  # optional metadata collector
└── Visualisation         # optional rendering (pygame)
```

Every slot has a `*Dummy` no-op implementation so components can be omitted cleanly.

### Key Design Patterns

**StEveObject base class** (`steve/core/steveobject.py`): every component inherits from this. It auto-generates YAML config from `__init__` parameters via `to_config()` / `from_config()` / `save_config()` / `from_config_file()`.

**ConfigHandler** (`steve/core/confighandler.py`): serializes/deserializes full component graphs to YAML, tracks shared object references by ID, supports "exchange" for swapping components when loading.

**Combination pattern**: `Reward`, `Terminal`, and `Truncation` all have `Combination` subclasses that compose multiple instances (e.g., `terminal.Combination([TargetReached(), MaxSteps(500)])`).

**Wrapper pattern**: `Observation` supports wrappers (e.g., `NormalizeTracking2DEpisode`) that wrap an inner observation and transform its output.

### Episode Flow

```
reset(seed):
  intervention.reset → start.reset → pathfinder.reset → interim_target.reset
  → observation.reset → reward.reset → terminal.reset → truncation.reset
  → info.reset → visualisation.reset

step(action):
  intervention.step(action) → pathfinder.step → interim_target.step
  → observation.step → reward.step → terminal.step → truncation.step
  → info.step
  returns: (obs, reward, terminated, truncated, info_dict)
```

### VesselTree & Branches

The primary vessel model is `AorticArch`. Branches are named: `lcca` (left common carotid), `rcca` (right common carotid), `lsa` (left subclavian), `rsa` (right subclavian), `bct` (brachiocephalic trunk), `co` (common origin). Branch definitions live in `steve/intervention/vesseltree/aorticarcharteries/` and `steve/intervention/vesseltree/branches/`.

### Simulation Backend

`SofaBeamAdapter` (`steve/intervention/simulation/sofabeamadapter/`) wraps SOFA v23.06+. There is also a `SimulationMP` multiprocessing wrapper. SOFA must be installed separately — it is not a pip dependency.

### Fluoroscopy

Provides 2D x-ray-like images or tracking coordinates. Implementations: `SimulatedFluoroscopy`, `TrackingOnly`, `Pillow`. The `Fluoroscopy` base defines the interface in `steve/intervention/fluoroscopy/`.

## Typical Usage Pattern

```python
vessel_tree = AorticArch(...)
instrument  = Angled(...)
simulation  = SofaBeamAdapter(vessel_tree, instrument)
fluoroscopy = SimulatedFluoroscopy(simulation)
target      = CenterlineRandom(vessel_tree)
intervention = MonoPlaneStatic(vessel_tree, [instrument], simulation, fluoroscopy, target)

observation = Tracking2D(fluoroscopy)
reward      = TargetReached(target)
terminal    = terminal.TargetReached(target)
truncation  = MaxSteps(500)

env = Env(intervention, observation, reward, terminal, truncation)
obs, info = env.reset()
obs, r, done, trunc, info = env.step(action)
```

See `examples/` for complete, runnable configurations.

## Documentation Style

This project follows the [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html).

**Module docstring** — every `.py` file opens with a one-line summary terminated by a period, optionally followed by a blank line and a longer description or usage example.

```python
"""A one-line summary of the module, terminated by a period.

Optional longer description. May include a usage example:

  foo = ClassFoo()
  bar = foo.function_bar()
"""
```

**Class docstring** — summary sentence(s) followed by an `Args:` block for constructor parameters and an `Attributes:` block for public instance attributes that are set after construction (e.g. populated by `reset()`). Do not repeat the type annotation in prose — the type hint is already in the signature. Constructor args go in the *class* docstring (not `__init__`).

```python
class Foo(StEveObject):
    """One-line summary.

    Longer explanation if needed.

    Args:
        param_a: What it controls.
        param_b: What it controls. Defaults to the vessel entry point
            when ``None``.

    Attributes:
        state: Current state, populated by ``reset()``.
    """
```

**Method docstring** — one-line summary. Add `Args:` / `Returns:` / `Raises:` sections only when the behaviour is non-obvious from the signature alone. Overriding methods only need a docstring if their behaviour materially differs from the base class.

```python
def step(self, action: np.ndarray) -> Tuple[...]:
    """Advance the environment by one timestep.

    Args:
        action: Velocity commands per instrument, shape ``(n, 2)``.

    Returns:
        Tuple of ``(observation, reward, terminated, truncated, info)``.

    Raises:
        SimulationError: If the physics backend reports a failure.
    """
```

**Property docstring** — single line, no `Returns:` section needed.

```python
@property
def observation_space(self) -> gym.Space:
    """The observation space of the environment."""
```

**Inline comments** — at least two spaces before the `#`; only for non-obvious invariants or workarounds, never restating what the code already says.

**Section dividers** — use light em-dash style, not heavy `# ---` banners:

```python
# — serialization —

# — public API —
```
