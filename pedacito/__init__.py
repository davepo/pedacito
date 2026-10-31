"""pedacito: an index-and-lookup intermediary for coding with a small local LLM.

Builds a searchable index of a project (AST-derived structure plus optional
LLM-generated summaries) so a model served by any OpenAI-compatible server can
retrieve only the relevant code to answer questions or propose edits, instead
of fitting the whole project in context.

See pedacito/cli.py for the command-line entry point and usage examples.
"""

__version__ = "0.1.0"
