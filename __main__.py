"""Fallback entry point: runs `python __main__.py ...` from an uninstalled
checkout, in addition to `python -m pedacito` and the installed `pedacito`
command."""

from pedacito.cli import main

raise SystemExit(main())
