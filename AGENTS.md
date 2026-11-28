# Agent Guidelines

## Testing Rules

- Always invoke pytest via `python -m pytest` (never bare `pytest`).
- During individual work packages, run only targeted tests (`python -m pytest tests/test_<module>.py -x -q`) or fast tests (`python -m pytest -m "not slow" -x -q`).
- Reserve the full suite (`python -m pytest`) strictly for the final release gate.
- Cap repair attempts at 2 per work package before pausing or escalating.
