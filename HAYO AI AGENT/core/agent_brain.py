"""
Unified Brain System for HAYO AI Agent

Integrates all 5 advanced systems:
  1. Memory System (short/medium/long-term context)
  2. Learning System (learn from executions)
  3. Planning System (analyze optimal paths)
  4. Dialogue System (natural conversation)
  5. Model Intelligence (Claude Opus for advanced reasoning)
"""

from __future__ import annotations

from pathlib import Path
from core.memory_system import MemorySystem, get_memory_context
from core.learning_system import LearningSystem
from core.planning_system import PlanningSystem, ExecutionPath
from core.dialogue_system import DialogueSystem, UserIntent

from langchain_core.messages import BaseMessage


class AgentBrain:
    """
    Unified brain that coordinates all intelligent systems.
    Each agent instance has one brain that persists across conversations.
    """

    _instance: AgentBrain | None = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if AgentBrain._initialized:
            return

        memory_dir = Path("./agent_memory")

        self.memory = MemorySystem(storage_dir=memory_dir)
        self.learning = LearningSystem(storage_dir=memory_dir)
        self.planning = PlanningSystem(learning_system=self.learning)
        self.dialogue = DialogueSystem()

        AgentBrain._initialized = True

    def add_message_to_memory(self, message: BaseMessage) -> None:
        """Add a message to the three-level memory system."""
        self.memory.add_message(message)

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
        self.learning.record_tool_execution(
            tool_name=tool_name,
            parameters=parameters,
            success=success,
            output=output,
            error_message=error_message,
            execution_time_ms=execution_time_ms,
        )

    def get_contextual_system_prompt(self, current_task: str) -> str:
        """
        Generate a dynamic system prompt that includes:
          - Current task context
          - Relevant past learning
          - Memory from short/medium-term
          - Learned preferences
        """
        memory_context = get_memory_context(
            self.memory,
            current_task,
            include_short=True,
            include_medium=True,
            include_long=True,
        )

        learning_report = self.learning.get_learning_report()

        prompt = f"""You are HAYO — an intelligent local OS agent for Windows.

CURRENT CONTEXT:
{memory_context}

LEARNED PATTERNS:
{learning_report}

CAPABILITIES:
• Intelligent reasoning and planning (Claude Opus level)
• Learn from execution history and past solutions
• Analyze multiple paths and choose optimal ones
• Natural, context-aware dialogue
• Persistent memory across sessions
• No artificial restrictions — unlimited execution

YOUR APPROACH:
1. Understand the task deeply using available context
2. Plan multiple approaches and pick the best one
3. Execute confidently and fully
4. Learn from outcomes for future improvements
5. Communicate naturally in the user's language
"""

        return prompt

    def analyze_task(
        self,
        task: str,
        candidate_paths: list[ExecutionPath],
        user_preferences: dict | None = None,
    ):
        """Analyze a task using the planning system."""
        return self.planning.analyze_task(task, candidate_paths, user_preferences)

    def classify_user_intent(self, message: str) -> UserIntent:
        """Classify what the user is asking."""
        return self.dialogue.classify_intent(message)

    def check_needs_clarification(self, message: str) -> str | None:
        """Check if we need to ask clarifying questions."""
        return self.dialogue.needs_clarification(message)

    def generate_natural_response(
        self,
        user_message: str,
        intent: UserIntent,
        execution_result: str | None = None,
        context_info: str | None = None,
    ) -> str:
        """Generate a natural response to the user."""
        return self.dialogue.generate_response(
            user_message,
            intent,
            execution_result,
            context_info,
        )

    def record_solution(
        self,
        problem: str,
        steps: list[str],
        tools_used: list[str],
        success: bool,
        context_tags: list[str] | None = None,
    ) -> None:
        """Record a solution for future reference."""
        self.learning.record_solution(problem, steps, tools_used, success, context_tags)

    def get_tool_reliability(self, tool_name: str) -> float:
        """Get the reliability score for a tool."""
        return self.learning.get_tool_reliability(tool_name)

    def get_best_solution_for_problem(
        self,
        problem: str,
        min_confidence: float = 0.6,
    ):
        """Get the best known solution for a problem."""
        return self.learning.get_best_solution_for_problem(problem, min_confidence)

    def record_preference(self, key: str, value: str) -> None:
        """Record a user preference."""
        self.learning.record_preference(key, value)

    def get_memory_stats(self) -> dict:
        """Get statistics about all memory systems."""
        return {
            "short_term_messages": len(self.memory.short_term),
            "medium_term_messages": len(self.memory.medium_term),
            "long_term_insights": len(self.memory.long_term_insights),
            "tool_reliability_records": len(self.learning.tool_success_rates),
            "learned_solutions": len(self.learning.learned_solutions),
            "execution_history": len(self.learning.execution_history),
            "user_preferences": len(self.learning.user_preferences),
        }


# Singleton instance
_brain = None


def get_brain() -> AgentBrain:
    """Get or create the singleton brain instance."""
    global _brain
    if _brain is None:
        _brain = AgentBrain()
    return _brain


def reset_brain_session() -> None:
    """Reset memory for a new session (but keep long-term learning)."""
    brain = get_brain()
    brain.memory.reset_session()
    brain.dialogue.context = brain.dialogue.__class__().context  # Reset dialogue context
