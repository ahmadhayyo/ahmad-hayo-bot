"""
Automatic Learning System for HAYO AI Agent

Learns from:
  - Tool execution outcomes (success/failure patterns)
  - Error messages and resolutions
  - User preferences and feedback
  - Optimal task execution paths

Storage:
  - Successes: Solutions that worked
  - Failures: Mistakes to avoid
  - Preferences: User preferences learned over time
  - Patterns: Recurring task types and optimal approaches
"""

from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict, field


@dataclass
class ExecutionOutcome:
    """Record of a tool execution and its outcome."""
    tool_name: str
    parameters: dict
    success: bool
    output: str
    error_message: str = ""
    execution_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    follow_up_actions: list[str] = field(default_factory=list)


@dataclass
class LearnedSolution:
    """A solution that worked for a specific problem type."""
    id: str
    problem_description: str
    steps: list[str]  # Ordered steps to solve
    tools_used: list[str]  # Tool names used in solution
    success_count: int  # How many times it worked
    failure_count: int  # How many times it failed
    success_rate: float  # success_count / (success_count + failure_count)
    last_used: float  # Timestamp
    context_tags: list[str]  # Categorization


class LearningSystem:
    """
    Automatically learns from execution history to improve future decisions.
    """

    def __init__(self, storage_dir: Path = Path("./agent_memory")):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

        # In-memory working copies
        self.execution_history: list[ExecutionOutcome] = []
        self.learned_solutions: list[LearnedSolution] = []
        self.error_patterns: dict[str, int] = {}  # Error message → frequency
        self.tool_success_rates: dict[str, tuple[int, int]] = {}  # tool_name → (success, total)
        self.user_preferences: dict[str, str] = {}  # Preference key → value

        self._load_from_disk()

    def record_tool_execution(
        self,
        tool_name: str,
        parameters: dict,
        success: bool,
        output: str,
        error_message: str = "",
        execution_time_ms: float = 0.0,
    ) -> None:
        """Record a tool execution for learning."""
        outcome = ExecutionOutcome(
            tool_name=tool_name,
            parameters=parameters,
            success=success,
            output=output,
            error_message=error_message,
            execution_time_ms=execution_time_ms,
        )

        self.execution_history.append(outcome)

        # Update success rates
        if tool_name not in self.tool_success_rates:
            self.tool_success_rates[tool_name] = (0, 0)

        success_count, total_count = self.tool_success_rates[tool_name]
        if success:
            success_count += 1
        total_count += 1
        self.tool_success_rates[tool_name] = (success_count, total_count)

        # Track error patterns
        if error_message:
            error_key = self._normalize_error(error_message)
            self.error_patterns[error_key] = self.error_patterns.get(error_key, 0) + 1

        # Keep history bounded
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-500:]

        self._save_to_disk()

    def record_solution(
        self,
        problem: str,
        steps: list[str],
        tools_used: list[str],
        success: bool,
        context_tags: list[str] | None = None,
    ) -> None:
        """Record a solution path that worked (or failed) for a problem."""
        solution_id = hashlib.md5(
            (problem + "".join(steps)).encode()
        ).hexdigest()[:12]

        # Check if we've seen this solution before
        existing = next((s for s in self.learned_solutions if s.id == solution_id), None)

        if existing:
            # Update statistics
            if success:
                existing.success_count += 1
            else:
                existing.failure_count += 1
            existing.success_rate = existing.success_count / (
                existing.success_count + existing.failure_count
            )
            existing.last_used = time.time()
        else:
            # Create new solution record
            solution = LearnedSolution(
                id=solution_id,
                problem_description=problem,
                steps=steps,
                tools_used=tools_used,
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                success_rate=1.0 if success else 0.0,
                last_used=time.time(),
                context_tags=context_tags or [],
            )
            self.learned_solutions.append(solution)

        # Keep solutions bounded (prune low-confidence old solutions)
        if len(self.learned_solutions) > 100:
            self.learned_solutions.sort(
                key=lambda s: (s.success_rate, s.success_count, s.last_used),
                reverse=True,
            )
            self.learned_solutions = self.learned_solutions[:75]

        self._save_to_disk()

    def get_tool_reliability(self, tool_name: str) -> float:
        """Get success rate (0.0-1.0) for a specific tool."""
        if tool_name not in self.tool_success_rates:
            return 0.5  # Unknown tools: neutral confidence

        success, total = self.tool_success_rates[tool_name]
        if total == 0:
            return 0.5

        return success / total

    def get_best_solution_for_problem(
        self,
        problem_description: str,
        min_confidence: float = 0.6,
    ) -> Optional[LearnedSolution]:
        """
        Find the best solution for a problem based on past successes.

        Returns:
            LearnedSolution with highest success rate, or None if none qualify.
        """
        # Simple keyword matching (could be enhanced with semantic similarity)
        candidates = [
            s
            for s in self.learned_solutions
            if any(
                keyword in problem_description.lower()
                for keyword in s.problem_description.lower().split()
                if len(keyword) > 3
            )
            and s.success_rate >= min_confidence
        ]

        if not candidates:
            return None

        # Return highest success rate
        return max(candidates, key=lambda s: (s.success_rate, s.success_count))

    def get_error_workaround(self, error_message: str) -> Optional[str]:
        """
        Get a known workaround for a specific error.
        Returns a workaround if one exists in the history.
        """
        error_key = self._normalize_error(error_message)

        # Find most recent successful recovery from this error
        for outcome in reversed(self.execution_history):
            if (
                error_key in self._normalize_error(outcome.error_message)
                and outcome.follow_up_actions
            ):
                # Found a recovery path
                return outcome.follow_up_actions[0]

        return None

    def record_preference(self, key: str, value: str) -> None:
        """Record a user preference learned during execution."""
        self.user_preferences[key] = value
        self._save_to_disk()

    def get_preference(self, key: str) -> Optional[str]:
        """Retrieve a learned user preference."""
        return self.user_preferences.get(key)

    def get_learning_report(self) -> str:
        """Generate a summary of what the system has learned."""
        parts = []

        # Tool statistics
        if self.tool_success_rates:
            parts.append("=== Tool Reliability ===")
            sorted_tools = sorted(
                self.tool_success_rates.items(),
                key=lambda x: x[1][0] / x[1][1] if x[1][1] > 0 else 0,
                reverse=True,
            )
            for tool_name, (success, total) in sorted_tools[:10]:
                rate = (success / total * 100) if total > 0 else 0
                parts.append(f"• {tool_name}: {success}/{total} ({rate:.0f}%)")

        # Common errors
        if self.error_patterns:
            parts.append("\n=== Common Errors ===")
            sorted_errors = sorted(
                self.error_patterns.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for error, count in sorted_errors[:5]:
                parts.append(f"• {error}: {count} times")

        # Best solutions
        if self.learned_solutions:
            parts.append("\n=== Best Solutions ===")
            best = sorted(
                self.learned_solutions,
                key=lambda s: (s.success_rate, s.success_count),
                reverse=True,
            )[:5]
            for solution in best:
                parts.append(
                    f"• {solution.problem_description[:50]}: {solution.success_rate:.0%} success"
                )

        # Learned preferences
        if self.user_preferences:
            parts.append(f"\n=== Learned Preferences ({len(self.user_preferences)}) ===")
            for key, value in list(self.user_preferences.items())[:5]:
                parts.append(f"• {key}: {value}")

        return "\n".join(parts) if parts else "No learning data yet."

    def _normalize_error(self, error_msg: str) -> str:
        """Normalize error message for pattern matching."""
        # Remove timestamps, file paths, and specific numbers
        import re

        normalized = re.sub(r"\d{1,}", "NUM", error_msg)
        normalized = re.sub(r"[a-zA-Z]:\\[^:]*", "PATH", normalized)
        normalized = normalized.lower()

        # Take first 100 chars to group similar errors
        return normalized[:100]

    def _save_to_disk(self) -> None:
        """Persist learning data to disk."""
        # Save execution history (keep last 500)
        history_file = self.storage_dir / "execution_history.json"
        history_data = [asdict(o) for o in self.execution_history[-500:]]
        history_file.write_text(json.dumps(history_data, indent=2))

        # Save learned solutions
        solutions_file = self.storage_dir / "learned_solutions.json"
        solutions_data = [asdict(s) for s in self.learned_solutions]
        solutions_file.write_text(json.dumps(solutions_data, indent=2))

        # Save metadata
        metadata = {
            "tool_success_rates": {
                k: {"success": v[0], "total": v[1]}
                for k, v in self.tool_success_rates.items()
            },
            "error_patterns": self.error_patterns,
            "user_preferences": self.user_preferences,
            "last_updated": time.time(),
        }
        metadata_file = self.storage_dir / "learning_metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))

    def _load_from_disk(self) -> None:
        """Load learning data from disk."""
        # Load execution history
        history_file = self.storage_dir / "execution_history.json"
        if history_file.exists():
            try:
                data = json.loads(history_file.read_text())
                self.execution_history = [ExecutionOutcome(**item) for item in data]
            except (json.JSONDecodeError, TypeError):
                self.execution_history = []

        # Load learned solutions
        solutions_file = self.storage_dir / "learned_solutions.json"
        if solutions_file.exists():
            try:
                data = json.loads(solutions_file.read_text())
                self.learned_solutions = [LearnedSolution(**item) for item in data]
            except (json.JSONDecodeError, TypeError):
                self.learned_solutions = []

        # Load metadata
        metadata_file = self.storage_dir / "learning_metadata.json"
        if metadata_file.exists():
            try:
                metadata = json.loads(metadata_file.read_text())
                self.tool_success_rates = {
                    k: (v["success"], v["total"])
                    for k, v in metadata.get("tool_success_rates", {}).items()
                }
                self.error_patterns = metadata.get("error_patterns", {})
                self.user_preferences = metadata.get("user_preferences", {})
            except (json.JSONDecodeError, TypeError):
                pass
