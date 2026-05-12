"""
Deduplication utilities for preventing repeated tool calls and duplicate messages.

Prevents:
  1. Calling the same tool with identical parameters multiple times
  2. Sending identical AI messages consecutively
  3. Tool call loops where the same action repeats endlessly
"""

from __future__ import annotations

import hashlib
import json
from typing import Any
from langchain_core.messages import AIMessage, BaseMessage


def _hash_content(content: str) -> str:
    """Create a hash of content for comparison."""
    return hashlib.md5(content.encode()).hexdigest()


def is_duplicate_tool_call(
    tool_name: str,
    tool_args: dict,
    tool_history: list[dict],
    recent_count: int = 2,
) -> bool:
    """
    Check if this tool has been called with identical args in the last N calls.

    Args:
        tool_name: Name of the tool being called
        tool_args: Arguments being passed
        tool_history: List of recent tool calls (from state)
        recent_count: How many recent calls to check (default 2)

    Returns:
        True if duplicate found, False otherwise
    """
    if not tool_history:
        return False

    # Get the last N tool calls
    recent_calls = tool_history[-recent_count:]

    for call in recent_calls:
        if call.get("name") == tool_name:
            # Compare args (normalize JSON for comparison)
            try:
                saved_args = call.get("args", {})
                if json.dumps(tool_args, sort_keys=True) == json.dumps(saved_args, sort_keys=True):
                    return True
            except (TypeError, ValueError):
                # If we can't compare, consider it different
                pass

    return False


def is_duplicate_message(
    new_message: BaseMessage,
    recent_messages: list[BaseMessage],
    min_length: int = 50,
) -> bool:
    """
    Check if this message is identical to a recent message.

    Args:
        new_message: The message to check
        recent_messages: List of recent messages to compare against (last 10)
        min_length: Minimum content length to consider (skip very short messages)

    Returns:
        True if duplicate found, False otherwise
    """
    if not isinstance(new_message, AIMessage):
        return False

    new_content = new_message.content
    if isinstance(new_content, str) and len(new_content) < min_length:
        return False

    new_hash = _hash_content(str(new_content))

    # Check against last 10 messages
    for msg in recent_messages[-10:]:
        if isinstance(msg, AIMessage):
            old_content = msg.content
            if isinstance(old_content, str):
                old_hash = _hash_content(old_content)
                if new_hash == old_hash:
                    return True

    return False


def record_tool_call(
    tool_name: str,
    tool_args: dict,
    result: str | None = None,
    tool_history: list[dict] | None = None,
    max_history: int = 20,
) -> list[dict]:
    """
    Record a tool call in the history (keeps last N calls).

    Args:
        tool_name: Name of the tool
        tool_args: Arguments passed
        result: The result (optional, for logging)
        tool_history: Existing history list (will be updated)
        max_history: Maximum calls to keep (default 20)

    Returns:
        Updated tool history (last N calls)
    """
    import time

    if tool_history is None:
        tool_history = []

    call_record = {
        "name": tool_name,
        "args": tool_args,
        "timestamp": time.time(),
        "result": result[:100] if result else None,  # Keep first 100 chars of result
    }

    # Add to history and keep only last N
    tool_history = list(tool_history) + [call_record]
    return tool_history[-max_history:]


def get_duplicate_prevention_status(
    state: dict,
) -> dict:
    """
    Analyze the state for duplicate patterns and return a summary.

    Returns:
        {
            "recent_tool": str,  # Last tool called
            "tool_calls_last_iteration": int,  # How many times called in last iteration
            "message_streak": int,  # How many identical messages in a row
            "is_at_risk": bool,  # True if patterns suggest looping
        }
    """
    tool_history = state.get("tool_call_history", [])
    messages = state.get("messages", [])
    last_tool = state.get("last_tool_name", "")

    # Count recent calls to same tool
    recent_count = 0
    if tool_history and last_tool:
        for call in tool_history[-5:]:
            if call.get("name") == last_tool:
                recent_count += 1

    # Count identical messages in a row
    message_streak = 0
    if len(messages) >= 2:
        for i in range(len(messages) - 1, 0, -1):
            if isinstance(messages[i], AIMessage) and isinstance(messages[i-1], AIMessage):
                if _hash_content(str(messages[i].content)) == _hash_content(str(messages[i-1].content)):
                    message_streak += 1
                else:
                    break

    return {
        "recent_tool": last_tool,
        "tool_calls_last_iteration": recent_count,
        "message_streak": message_streak,
        "is_at_risk": recent_count >= 2 or message_streak >= 2,
    }
