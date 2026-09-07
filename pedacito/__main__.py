"""Fallback entry point: allows running `python __main__.py ...` directly
from an uninstalled checkout of this repository, in addition to the
`python -m pedacito` and installed `pedacito` command forms (see
pedacito/__main__.py and pyproject.toml's [project.scripts])."""

from pedacito.cli import main

raise SystemExit(main())
