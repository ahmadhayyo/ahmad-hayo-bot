"""
agent/workflow.py - LangGraph StateGraph compilation with persistent memory.

Checkpointer selection:
  1. AsyncSqliteSaver (preferred, supports Chainlit's astream_events)
  2. SqliteSaver (sync-only fallback)
  3. MemorySaver (last resort, no persistence)
"""

import logging
import os

from langgraph.graph import END, StateGraph

from agent.nodes import (
    planner_node,
    reviewer_node,
    should_continue,
    worker_node,
)
from core.state import AgentState

logger = logging.getLogger("hayo.workflow")

_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "agent_memory.db"
)

_COMPILED_GRAPH = None


def _build_checkpointer():
    """Pick the best available checkpointer."""
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        import aiosqlite

        conn = aiosqlite.connect(_DB_PATH, check_same_thread=False)
        logger.info("Persistent memory (async): %s", _DB_PATH)
        return AsyncSqliteSaver(conn)
    except ImportError as exc:
        logger.warning("AsyncSqliteSaver unavailable (%s). pip install aiosqlite", exc)
    except Exception as exc:
        logger.warning("AsyncSqliteSaver init failed (%s)", exc)

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3

        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        logger.warning("Persistent memory (sync only): %s", _DB_PATH)
        return SqliteSaver(conn)
    except Exception as exc:
        logger.warning("SqliteSaver unavailable (%s)", exc)

    from langgraph.checkpoint.memory import MemorySaver
    logger.warning("Using MemorySaver — no persistence between restarts")
    return MemorySaver()


def compile_graph():
    """Build and compile the agent StateGraph (singleton)."""
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is not None:
        return _COMPILED_GRAPH

    builder = StateGraph(AgentState)
    builder.add_node("planner", planner_node)
    builder.add_node("worker", worker_node)
    builder.add_node("reviewer", reviewer_node)
    builder.set_entry_point("planner")
    builder.add_edge("planner", "worker")
    builder.add_edge("worker", "reviewer")
    builder.add_conditional_edges(
        "reviewer",
        should_continue,
        {"worker": "worker", "__end__": END},
    )

    _COMPILED_GRAPH = builder.compile(checkpointer=_build_checkpointer())
    logger.info("Agent graph compiled successfully")
    return _COMPILED_GRAPH
