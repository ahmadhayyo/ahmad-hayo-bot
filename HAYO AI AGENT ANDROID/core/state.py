"""
AgentState — Shared state schema for HAYO AI Agent (Android Edition).
"""

from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    plan: list[str]
    completed_steps: list[str]
    workspace: str
    error_logs: list[str]
    iteration_count: int
    requires_human_approval: bool
    pending_command: str
