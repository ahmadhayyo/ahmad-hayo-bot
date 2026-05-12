"""
Three-Level Memory System for HAYO AI Agent

Architecture:
  - SHORT-TERM (current session): Last 20 messages, immediate context
  - MEDIUM-TERM (recent history): Last 100 messages, task context over ~30 min
  - LONG-TERM (learning): Extracted insights, patterns, solutions from all sessions

Strategy:
  1. All messages flow to SHORT-TERM first
  2. When SHORT-TERM fills, oldest move to MEDIUM-TERM with consolidation
  3. When MEDIUM-TERM fills, insights are extracted and stored as LONG-TERM
  4. During planning/execution, relevant LONG-TERM insights are injected as context
"""

from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage


@dataclass
class MemoryInsight:
    """A condensed insight extracted from multiple messages."""
    id: str
    timestamp: float
    category: str  # "tool_usage", "error_pattern", "solution", "preference"
    summary: str
    relevant_context: str
    confidence: float  # 0.0-1.0
    frequency: int  # How many times seen
    related_tasks: list[str]  # Task types this applies to


class MemorySystem:
    """
    Three-level memory management:
      - SHORT_TERM: Current session's working memory (20 messages)
      - MEDIUM_TERM: Recent context for this task (100 messages)
      - LONG_TERM: Persistent insights across sessions
    """

    def __init__(self, storage_dir: Path = Path("./agent_memory")):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        # In-memory working copies
        self.short_term: list[BaseMessage] = []  # Last 20 messages
        self.medium_term: list[BaseMessage] = []  # Last 100 messages
        self.long_term_insights: list[MemoryInsight] = []  # Persistent learnings

        # Configuration
        self.SHORT_TERM_LIMIT = 20
        self.MEDIUM_TERM_LIMIT = 100
        self.LONG_TERM_INSIGHT_LIMIT = 50

        self._load_long_term_from_disk()

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to short-term memory, cascading to lower levels as needed."""
        self.short_term.append(message)

        # When short-term overflows, push to medium-term
        if len(self.short_term) > self.SHORT_TERM_LIMIT:
            overflow = self.short_term[0]
            self.short_term = self.short_term[1:]
            self.medium_term.append(overflow)

        # When medium-term overflows, extract insights to long-term
        if len(self.medium_term) > self.MEDIUM_TERM_LIMIT:
            overflow = self.medium_term[0]
            self.medium_term = self.medium_term[1:]
            self._extract_insight_if_valuable(overflow)

    def get_short_term(self) -> list[BaseMessage]:
        """Get current working memory (last 20 messages)."""
        return list(self.short_term)

    def get_medium_term(self) -> list[BaseMessage]:
        """Get recent context (last 100 messages)."""
        return list(self.medium_term)

    def get_relevant_long_term(self, query: str, limit: int = 5) -> list[MemoryInsight]:
        """
        Retrieve relevant long-term insights based on query keywords.

        Args:
            query: Task description or keywords to match against insights
            limit: Max insights to return

        Returns:
            Sorted list of most relevant insights
        """
        # Simple relevance scoring based on keyword overlap
        query_words = set(query.lower().split())

        scored = []
        for insight in self.long_term_insights:
            insight_words = set(
                (insight.summary + " " + insight.relevant_context).lower().split()
            )
            overlap = len(query_words & insight_words)

            if overlap > 0:
                # Score: overlap count + confidence + frequency bonus
                score = overlap + (insight.confidence * 10) + (insight.frequency * 0.5)
                scored.append((score, insight))

        # Return top N by score, highest first
        scored.sort(reverse=True, key=lambda x: x[0])
        return [insight for _, insight in scored[:limit]]

    def _extract_insight_if_valuable(self, message: BaseMessage) -> None:
        """
        Extract a learning insight from a message if it contains valuable information.
        Examples: tool errors, solutions, patterns, preferences.
        """
        content = message.content if isinstance(message.content, str) else ""

        # Categorize the message type
        if isinstance(message, ToolMessage):
            category = "tool_execution"
            if "error" in content.lower() or "failed" in content.lower():
                category = "error_pattern"
        elif isinstance(message, AIMessage):
            if "solution" in content.lower() or "fixed" in content.lower():
                category = "solution"
            else:
                category = "reasoning"
        elif isinstance(message, HumanMessage):
            category = "user_preference"
        else:
            return  # Skip system messages

        # Only create insight if message is substantial
        if len(content) < 20:
            return

        # Create insight
        insight_id = hashlib.md5(
            (content + str(time.time())).encode()
        ).hexdigest()[:12]

        insight = MemoryInsight(
            id=insight_id,
            timestamp=time.time(),
            category=category,
            summary=content[:200],  # First 200 chars
            relevant_context=content[:500],  # Full message up to 500 chars
            confidence=0.7,  # Start with medium confidence
            frequency=1,
            related_tasks=self._extract_task_tags(content),
        )

        self.long_term_insights.append(insight)

        # Keep long-term under limit
        if len(self.long_term_insights) > self.LONG_TERM_INSIGHT_LIMIT:
            # Remove least valuable (lowest confidence + lowest frequency)
            self.long_term_insights.sort(
                key=lambda x: x.confidence * x.frequency
            )
            self.long_term_insights = self.long_term_insights[1:]

        # Save to disk
        self._save_long_term_to_disk()

    def _extract_task_tags(self, content: str) -> list[str]:
        """Extract task category tags from message content."""
        tags = []
        content_lower = content.lower()

        # Simple pattern matching
        if any(word in content_lower for word in ["open", "app", "launch", "start"]):
            tags.append("app_management")
        if any(word in content_lower for word in ["download", "upload", "file"]):
            tags.append("file_operations")
        if any(word in content_lower for word in ["search", "find", "lookup"]):
            tags.append("search")
        if any(word in content_lower for word in ["excel", "csv", "data"]):
            tags.append("data_processing")
        if any(word in content_lower for word in ["error", "failed", "problem"]):
            tags.append("troubleshooting")

        return tags

    def _save_long_term_to_disk(self) -> None:
        """Persist long-term insights to disk."""
        insights_file = self.storage_dir / "long_term_insights.json"

        data = [asdict(insight) for insight in self.long_term_insights]
        insights_file.write_text(json.dumps(data, indent=2))

    def _load_long_term_from_disk(self) -> None:
        """Load long-term insights from disk."""
        insights_file = self.storage_dir / "long_term_insights.json"

        if insights_file.exists():
            try:
                data = json.loads(insights_file.read_text())
                self.long_term_insights = [
                    MemoryInsight(**item) for item in data
                ]
            except (json.JSONDecodeError, TypeError):
                self.long_term_insights = []

    def reset_session(self) -> None:
        """Reset short and medium-term memory (for new session/task)."""
        self.short_term = []
        self.medium_term = []
        # Long-term persists across sessions

    def clear_all(self) -> None:
        """Complete memory wipe (use cautiously)."""
        self.short_term = []
        self.medium_term = []
        self.long_term_insights = []
        insights_file = self.storage_dir / "long_term_insights.json"
        if insights_file.exists():
            insights_file.unlink()


def get_memory_context(
    memory: MemorySystem,
    current_task: str,
    include_short: bool = True,
    include_medium: bool = True,
    include_long: bool = True,
) -> str:
    """
    Build a context string from all three memory levels.
    Use this to inject memory into the system prompt or tool context.
    """
    parts = []

    if include_short and memory.short_term:
        parts.append("=== Current Context (Last 20 messages) ===")
        for msg in memory.short_term[-5:]:  # Last 5 for brevity
            parts.append(f"• {msg.__class__.__name__}: {str(msg.content)[:100]}")

    if include_medium and memory.medium_term:
        parts.append("\n=== Recent Task Context ===")
        parts.append(
            f"(Context from {len(memory.medium_term)} recent messages available)"
        )

    if include_long:
        relevant = memory.get_relevant_long_term(current_task, limit=3)
        if relevant:
            parts.append("\n=== Relevant Past Learning ===")
            for insight in relevant:
                parts.append(
                    f"• [{insight.category}] {insight.summary}\n"
                    f"  Confidence: {insight.confidence:.1%}, Frequency: {insight.frequency}x"
                )

    return "\n".join(parts)
