"""
Shim — kept for backwards compatibility with code that imported build_graph.

The canonical path is `agent.workflow.compile_graph`. This module forwards.
"""

from __future__ import annotations

from agent.workflow import compile_graph


def build_graph(use_checkpointer: bool = True):  # noqa: ARG001
    """Backwards-compat alias. The flag is ignored — workflow always checkpoints."""
    return compile_graph()
