"""
agent/workflow.py — LangGraph StateGraph compilation with persistent memory.
"""

import os
from langgraph.graph import END, StateGraph
from agent.nodes import planner_node, reviewer_node, should_continue, worker_node
from core.state import AgentState

_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent_memory.db")
_COMPILED_GRAPH = None


def _build_checkpointer():
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        import aiosqlite
        conn = aiosqlite.connect(_DB_PATH, check_same_thread=False)
        print("[OK] Persistent memory (async):", _DB_PATH)
        return AsyncSqliteSaver(conn)
    except ImportError as exc:
        print("[WARN] AsyncSqliteSaver unavailable (%s)" % exc)
    except Exception as exc:
        print("[WARN] AsyncSqliteSaver init failed (%s)" % exc)

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        print("[WARN] Persistent memory (sync only):", _DB_PATH)
        return SqliteSaver(conn)
    except Exception as exc:
        print("[WARN] SqliteSaver unavailable (%s)" % exc)

    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()


def compile_graph():
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
        "reviewer", should_continue,
        {"worker": "worker", "__end__": END},
    )

    _COMPILED_GRAPH = builder.compile(checkpointer=_build_checkpointer())
    return _COMPILED_GRAPH
