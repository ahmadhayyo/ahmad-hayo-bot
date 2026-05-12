"""
Smart Planning System for HAYO AI Agent

Analyzes multiple execution paths and chooses the optimal one based on:
  - Tool reliability (from learning system)
  - Execution cost (time, complexity)
  - Risk assessment (potential failures)
  - User preferences and past patterns
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class ExecutionPath:
    """A possible way to accomplish a task."""
    name: str  # e.g., "Direct API Call", "Browser Automation", "CLI Tool"
    steps: list[str]  # Ordered steps
    tools: list[str]  # Tools needed
    estimated_time_ms: int  # Rough estimate
    complexity: str  # "low", "medium", "high"
    risk_level: str  # "low", "medium", "high" (chance of failure)
    estimated_reliability: float  # 0.0-1.0 based on tool success rates
    advantages: list[str]
    disadvantages: list[str]
    requires_approval: bool  # Destructive or sensitive operation?


@dataclass
class AnalysisResult:
    """Result of analyzing multiple execution paths."""
    task_description: str
    paths: list[ExecutionPath]
    recommended_path: Optional[ExecutionPath]
    recommendation_confidence: float  # 0.0-1.0
    analysis_notes: str


class PlanningSystem:
    """
    Intelligently analyzes multiple ways to accomplish a task and picks the best one.
    """

    def __init__(self, learning_system=None):
        self.learning_system = learning_system
        self.path_success_history: dict[str, list[bool]] = {}

    def analyze_task(
        self,
        task: str,
        candidate_paths: list[ExecutionPath],
        user_preferences: dict | None = None,
    ) -> AnalysisResult:
        """
        Analyze multiple execution paths and recommend the best one.

        Args:
            task: Task description
            candidate_paths: List of possible execution paths
            user_preferences: User preferences to factor in (e.g., "prefer_speed")

        Returns:
            AnalysisResult with recommended path and reasoning
        """
        user_prefs = user_preferences or {}

        # Score each path
        scored_paths = []
        for path in candidate_paths:
            score = self._score_path(path, user_prefs)
            scored_paths.append((score, path))

        # Sort by score (highest first)
        scored_paths.sort(reverse=True, key=lambda x: x[0])

        if not scored_paths:
            return AnalysisResult(
                task_description=task,
                paths=candidate_paths,
                recommended_path=None,
                recommendation_confidence=0.0,
                analysis_notes="No valid paths found.",
            )

        best_score, best_path = scored_paths[0]
        second_score = scored_paths[1][0] if len(scored_paths) > 1 else 0

        # Confidence = how much better the best path is than the second best
        confidence = self._calculate_confidence(best_score, second_score)

        analysis_notes = self._generate_analysis_notes(
            task,
            best_path,
            candidate_paths,
            scored_paths,
        )

        return AnalysisResult(
            task_description=task,
            paths=candidate_paths,
            recommended_path=best_path,
            recommendation_confidence=confidence,
            analysis_notes=analysis_notes,
        )

    def _score_path(self, path: ExecutionPath, user_prefs: dict) -> float:
        """Calculate a composite score for a path based on multiple factors."""
        score = 0.0

        # Base reliability (from learning system)
        if self.learning_system:
            path_reliability = self._estimate_path_reliability(path)
        else:
            path_reliability = path.estimated_reliability

        score += path_reliability * 40  # Reliability is most important

        # Factor in user preferences
        if user_prefs.get("prefer_speed"):
            # Shorter time = higher score (0-10 points)
            time_score = max(0, 10 - (path.estimated_time_ms / 1000))
            score += time_score

        if user_prefs.get("prefer_safety"):
            # Lower risk = higher score
            risk_penalty = {"low": 0, "medium": -5, "high": -10}.get(
                path.risk_level, -5
            )
            score += risk_penalty

        # Complexity penalty
        complexity_penalty = {"low": 0, "medium": -2, "high": -5}.get(
            path.complexity, 0
        )
        score += complexity_penalty

        # Past success rate for this path
        if path.name in self.path_success_history:
            successes = sum(self.path_success_history[path.name])
            total = len(self.path_success_history[path.name])
            if total > 0:
                historical_rate = successes / total
                score += historical_rate * 15

        return score

    def _estimate_path_reliability(self, path: ExecutionPath) -> float:
        """Estimate the overall reliability of a path based on its tools."""
        if not self.learning_system or not path.tools:
            return path.estimated_reliability

        # Average reliability of all tools in the path
        tool_reliabilities = []
        for tool in path.tools:
            reliability = self.learning_system.get_tool_reliability(tool)
            tool_reliabilities.append(reliability)

        if not tool_reliabilities:
            return path.estimated_reliability

        avg_reliability = sum(tool_reliabilities) / len(tool_reliabilities)

        # Slightly reduce reliability based on number of steps (more steps = more failure points)
        step_penalty = 0.98 ** (len(path.steps) - 1)

        return avg_reliability * step_penalty

    def _calculate_confidence(self, best_score: float, second_score: float) -> float:
        """
        Calculate confidence in recommendation.
        Higher difference = higher confidence.
        """
        if second_score == 0:
            return 1.0

        # Confidence is the ratio of difference to best score
        confidence = (best_score - second_score) / (best_score + 1)
        return min(1.0, max(0.0, confidence))

    def _generate_analysis_notes(
        self,
        task: str,
        best_path: ExecutionPath,
        all_paths: list[ExecutionPath],
        scored_paths: list[tuple[float, ExecutionPath]],
    ) -> str:
        """Generate human-readable analysis notes."""
        notes = []

        notes.append(f"TASK: {task}")
        notes.append(f"\nRECOMMENDED: {best_path.name}")
        notes.append(f"  Estimated time: {best_path.estimated_time_ms}ms")
        notes.append(f"  Complexity: {best_path.complexity}")
        notes.append(f"  Risk level: {best_path.risk_level}")
        notes.append(f"  Reliability: {best_path.estimated_reliability:.0%}")

        if best_path.advantages:
            notes.append("\n  Advantages:")
            for adv in best_path.advantages:
                notes.append(f"    + {adv}")

        if best_path.disadvantages:
            notes.append("\n  Disadvantages:")
            for dis in best_path.disadvantages:
                notes.append(f"    - {dis}")

        # Show alternatives
        if len(scored_paths) > 1:
            notes.append("\nALTERNATIVES:")
            for score, path in scored_paths[1:4]:  # Top 3 alternatives
                notes.append(
                    f"  • {path.name} (score: {score:.1f}, reliability: {path.estimated_reliability:.0%})"
                )

        return "\n".join(notes)

    def record_outcome(self, path_name: str, success: bool) -> None:
        """Record whether a path succeeded or failed for future analysis."""
        if path_name not in self.path_success_history:
            self.path_success_history[path_name] = []

        self.path_success_history[path_name].append(success)

        # Keep history bounded
        if len(self.path_success_history[path_name]) > 100:
            self.path_success_history[path_name] = self.path_success_history[path_name][
                -50:
            ]


def generate_execution_paths(task: str) -> list[ExecutionPath]:
    """
    Generate candidate execution paths for a given task.
    This is a template that should be customized per task type.
    """
    paths = []

    # Generic paths that apply to many tasks
    if "open" in task.lower() and "app" in task.lower():
        paths.append(
            ExecutionPath(
                name="Direct App Launch",
                steps=["Parse app name", "Use open_app tool"],
                tools=["open_app"],
                estimated_time_ms=2000,
                complexity="low",
                risk_level="low",
                estimated_reliability=0.95,
                advantages=["Fast", "Simple", "Direct"],
                disadvantages=["May not handle complex launch scenarios"],
                requires_approval=False,
            )
        )
        paths.append(
            ExecutionPath(
                name="Taskbar/Search",
                steps=["Open Windows search", "Type app name", "Press Enter"],
                tools=["keyboard_hotkey", "keyboard_type"],
                estimated_time_ms=3000,
                complexity="low",
                risk_level="low",
                estimated_reliability=0.85,
                advantages=["Uses native Windows", "User-like behavior"],
                disadvantages=["Slightly slower", "More interactions"],
                requires_approval=False,
            )
        )

    if "download" in task.lower() and "file" in task.lower():
        paths.append(
            ExecutionPath(
                name="Direct URL Download",
                steps=["Extract URL", "Use download_file tool"],
                tools=["download_file"],
                estimated_time_ms=5000,
                complexity="low",
                risk_level="low",
                estimated_reliability=0.90,
                advantages=["Fast", "Direct", "Reliable"],
                disadvantages=["Requires direct URL"],
                requires_approval=False,
            )
        )
        paths.append(
            ExecutionPath(
                name="Browser Download",
                steps=[
                    "Open browser",
                    "Navigate to site",
                    "Click download",
                    "Wait for completion",
                ],
                tools=["open_app", "browser_navigate", "wait"],
                estimated_time_ms=15000,
                complexity="medium",
                risk_level="medium",
                estimated_reliability=0.80,
                advantages=["Works with indirect downloads", "Handles auth"],
                disadvantages=["Slower", "More interactions", "Needs browser"],
                requires_approval=False,
            )
        )

    return paths


# Example task-specific analysis function
def analyze_download_task(
    url: str,
    destination: str,
    learning_system=None,
) -> AnalysisResult:
    """Analyze how best to download a file."""
    task = f"Download file from {url} to {destination}"

    # Generate candidate paths
    paths = generate_execution_paths(task)

    # Create planning system
    planner = PlanningSystem(learning_system=learning_system)

    # Analyze
    result = planner.analyze_task(task, paths)

    return result
