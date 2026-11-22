🌐 **English** | [Español](es/CONTRIBUTING.md)

# Contributing

Thanks for helping. Issues and pull requests are welcome; please open an issue before a large
PR. Report security problems privately, as described in
[SECURITY.md](SECURITY.md#reporting-a-vulnerability).

## Development environment

```bash
git clone https://github.com/davepo/pedacito && cd pedacito
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

```bash
pytest -q                       # the full suite
pytest -q tests/test_doc_sync.py tests/test_release_hygiene.py   # docs and release guards
ruff check pedacito tests       # lint
```

Python 3.11 or newer is required. Tests must not need a running model server.

## Codebase map

| Path | Role |
|---|---|
| `pedacito/cli.py` | Argument parser and every `pedacito` subcommand |
| `pedacito/config.py` | The `Config` dataclass, defaults, TOML loading, presets |
| `pedacito/index.py`, `chunker*.py` | Project parsing into units with exact line ranges |
| `pedacito/agent.py`, `loop.py`, `tools.py` | Gather loop, actions and the answer phase |
| `pedacito/edits.py`, `verify.py` | Edit blocks, repair and verification |
| `pedacito/llm.py`, `providers.py`, `servers.py` | Model clients, hosted provider table, server lifecycle |
| `pedacito/security.py`, `safety.py`, `sandbox.py` | Secret scanning, redaction, OS sandbox |
| `pedacito/skills.py`, `mcp.py`, `hooks.py`, `prompts.py` | Skills, MCP bridge, hooks, prompt overrides |
| `pedacito/tui.py`, `pedacito/gui/` | Terminal UI and browser control panel |
| `pedacito/i18n.py`, `pedacito/locales/` | Localization engine and catalogs |
| `pedacito/how_index.py`, `how.py` | The searchable `pedacito how` index |
| `tests/` | The test suite, including documentation drift guards |

## Documentation ownership

- `docs/` is the canonical **reference** for settings, commands and guarantees.
- The root guides ([QUICKSTART.md](../QUICKSTART.md), [MODEL_GUIDE.md](../MODEL_GUIDE.md),
  [TECHNIQUES.md](../TECHNIQUES.md)) are the canonical **walkthroughs**.
- [README.md](../README.md) is the index and [CHANGELOG.md](../CHANGELOG.md) is the history.
- Link between sets instead of copying text. The `**Docs ownership.**` paragraph in the README
  and in [ARCHITECTURE.md](ARCHITECTURE.md) must stay identical.
- A new setting goes into [CONFIG.md](CONFIG.md) with its real default, a new subcommand into
  the table in [USAGE.md](USAGE.md#the-cli-at-a-glance) and into `pedacito/how_index.py`.
  `tests/test_doc_sync.py` fails when they drift.

## Adding or maintaining a locale

1. Copy `pedacito/locales/en.json` to `pedacito/locales/<code>.json` and translate the values;
   keep every key and every `{placeholder}`.
2. Register the code in `i18n.SUPPORTED_LANGUAGES` and the `language` enum in `config.py`.
3. Translate the docs into `docs/<code>/` (the root guides become `README.<code>.md`,
   `QUICKSTART.<code>.md`, and so on) and add the matching language-switcher bar to the top
   of each English page.
4. List the new files in `[tool.setuptools.data-files]` of `pyproject.toml`.
5. Run `pytest -q`; the locale and documentation guards must stay green.

The Spanish catalog and docs (`pedacito/locales/es.json`, `docs/es/`) are the reference
example.
