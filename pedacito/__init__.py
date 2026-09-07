"""pedacito: an index-and-lookup intermediary for coding with a local LLM.

Builds a searchable index of a project (AST-derived structure plus optional
LLM-generated summaries) and lets a local model, run through LM Studio,
retrieve only the relevant pieces of a codebase to answer questions or
propose edits -- rather than requiring the whole project to fit in context.

See pedacito/cli.py for the command-line entry point and usage examples.
"""

__version__ = "0.1.0"
