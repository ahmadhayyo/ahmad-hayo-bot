"""
Advanced Natural Dialogue System for HAYO AI Agent

Features:
  - Intent recognition and classification
  - Context-aware conversation flow
  - Clarification questions for ambiguous requests
  - Personality and tone adaptation
  - Multi-turn conversation handling
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import re


class UserIntent(Enum):
    """Classification of user intent."""
    DIRECT_TASK = "direct_task"  # "Open chrome and download file.pdf"
    CLARIFICATION = "clarification"  # "How would you do that?"
    FEEDBACK = "feedback"  # "That didn't work" / "Perfect!"
    PREFERENCE = "preference"  # "I prefer faster methods"
    QUESTION = "question"  # "Can you do X?"
    GREETING = "greeting"  # "Hi", "Hello"
    CASUAL_CHAT = "casual_chat"  # Conversation, not a task
    CORRECTION = "correction"  # "No, not that. Try this."


@dataclass
class DialogueContext:
    """Track conversation context for coherent multi-turn dialogue."""
    current_task: Optional[str] = None  # What are we currently working on?
    previous_task: Optional[str] = None  # What did we just finish?
    pending_clarification: Optional[str] = None  # Question user asked us
    user_preferences: dict[str, str] | None = None  # Learned preferences
    conversation_history: list[tuple[str, str]] | None = None  # (user, assistant) pairs
    turn_count: int = 0  # How many exchanges in this conversation


class DialogueSystem:
    """
    Manages natural, context-aware conversations with the user.
    """

    def __init__(self):
        self.context = DialogueContext()
        self.clarification_patterns = {
            UserIntent.QUESTION: [
                r"^(can|could|would|will|would you).*\?$",
                r"^how\s+",
                r"^what.*\?$",
            ],
            UserIntent.PREFERENCE: [
                r"\bprefer",
                r"\brather",
                r"\binstead",
                r"\bi.*like.*better",
            ],
            UserIntent.FEEDBACK: [
                r"(great|perfect|thanks|that works)",
                r"(didn't work|failed|error|problem)",
                r"(wrong|incorrect|not.*right)",
            ],
            UserIntent.CORRECTION: [
                r"^no[,\.]?\s+(try|do|use|different)",
                r"^not\s+that",
                r"^instead",
            ],
        }

    def classify_intent(self, user_message: str) -> UserIntent:
        """Classify what the user is asking for."""
        msg_lower = user_message.lower().strip()

        # Greetings
        greeting_words = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon"]
        if any(msg_lower.startswith(word) for word in greeting_words):
            return UserIntent.GREETING

        # Check patterns
        for intent, patterns in self.clarification_patterns.items():
            for pattern in patterns:
                if re.search(pattern, msg_lower):
                    if intent == UserIntent.FEEDBACK:
                        if "didn't work" in msg_lower or "error" in msg_lower:
                            return intent
                        elif any(w in msg_lower for w in ["great", "perfect", "thanks"]):
                            return intent
                    else:
                        return intent

        # If it contains task-like verbs, it's a task
        task_verbs = [
            "open", "close", "download", "upload", "create", "delete",
            "convert", "find", "search", "show", "display", "send",
            "move", "copy", "rename", "read", "write"
        ]
        if any(msg_lower.startswith(verb) or f" {verb} " in msg_lower for verb in task_verbs):
            return UserIntent.DIRECT_TASK

        # Short casual messages
        if len(msg_lower.split()) <= 3:
            return UserIntent.CASUAL_CHAT

        # Default to task (assume user wants something done)
        return UserIntent.DIRECT_TASK

    def needs_clarification(self, message: str) -> Optional[str]:
        """
        Detect if a request is ambiguous and return clarification question.
        Returns a clarification question or None if clear enough.
        """
        msg_lower = message.lower()
        length = len(message.split())

        # Very short messages often need clarification
        if length <= 2:
            return f"I understand you want to work with something. Could you provide more details about what specifically?"

        # Check for vague pronouns or references
        if "it" in msg_lower or "that" in msg_lower:
            if self.context.current_task:
                # We have context, OK to proceed
                return None
            else:
                # No context, ambiguous
                return "I want to make sure I understand - what are we working with?"

        # Check for multiple possible interpretations
        if "or" in msg_lower:
            # Multiple options - ask user which
            parts = message.split(" or ")
            if len(parts) == 2:
                return f"Which would you prefer: {parts[0].strip()} or {parts[1].strip()}?"

        # Check for unclear file references
        if ("file" in msg_lower or "document" in msg_lower) and not (".pdf" in msg_lower or ".docx" in msg_lower or ".xlsx" in msg_lower):
            return "What file should I work with? Could you provide the filename or path?"

        # Check for unspecified locations
        if ("save" in msg_lower or "download" in msg_lower) and "to" not in msg_lower:
            return "Where should I save/download this? (e.g., Desktop, Downloads, specific path)"

        return None

    def generate_response(
        self,
        user_message: str,
        intent: UserIntent,
        execution_result: Optional[str] = None,
        context_info: Optional[str] = None,
    ) -> str:
        """
        Generate a contextually appropriate response.

        Args:
            user_message: What the user said
            intent: Classified intent
            execution_result: Result of executing a task (if applicable)
            context_info: Relevant context from memory system

        Returns:
            Assistant response
        """
        self.context.turn_count += 1

        if intent == UserIntent.GREETING:
            greetings = [
                "السلام عليكم! كيف يمكنني مساعدتك؟" if any(ord(c) > 127 for c in user_message) else "Hello! How can I help?",
                "مرحبا! ما الذي تود أن تفعل؟" if any(ord(c) > 127 for c in user_message) else "Hi there! What would you like me to do?",
            ]
            return greetings[self.context.turn_count % len(greetings)]

        if intent == UserIntent.CASUAL_CHAT:
            return "I'm here to help with tasks on your computer. What would you like me to do?"

        if intent == UserIntent.FEEDBACK:
            if "great" in user_message.lower() or "perfect" in user_message.lower():
                return "Great! I'm glad that worked. What else would you like me to do?"
            else:
                return "I see there was an issue. Let me try a different approach."

        if intent == UserIntent.QUESTION:
            # Try to answer factual questions
            if "can you" in user_message.lower():
                return "Yes, I can help with that. I have access to many tools and applications on your computer. Let me know what you need."
            return "Good question. What specifically would you like me to help with?"

        if intent == UserIntent.CORRECTION:
            return "Got it, I'll try a different approach. What should I do instead?"

        if intent == UserIntent.PREFERENCE:
            return "I'll remember your preference. Let me adjust my strategy accordingly."

        # Default: Direct task
        if execution_result:
            # Task was executed, acknowledge result
            if "error" in execution_result.lower() or "failed" in execution_result.lower():
                return f"I encountered an issue: {execution_result[:200]}. Let me try another approach."
            else:
                return f"Done! {execution_result[:200]}. What's next?"
        else:
            # Starting a new task
            self.context.current_task = user_message
            return f"I'll help you with that. Starting now..."

    def update_context(
        self,
        user_message: str,
        assistant_response: str,
        task_result: Optional[str] = None,
    ) -> None:
        """Update dialogue context after an exchange."""
        if not self.context.conversation_history:
            self.context.conversation_history = []

        self.context.conversation_history.append((user_message, assistant_response))

        # Keep history bounded
        if len(self.context.conversation_history) > 20:
            self.context.conversation_history = self.context.conversation_history[-15:]

        # Update task tracking
        intent = self.classify_intent(user_message)
        if intent == UserIntent.DIRECT_TASK:
            self.context.previous_task = self.context.current_task
            self.context.current_task = user_message

    def get_contextual_greeting(self) -> str:
        """Generate a context-aware greeting."""
        if self.context.previous_task:
            return f"Ready for the next task. (Previous: {self.context.previous_task[:50]}...)"

        if self.context.turn_count == 0:
            return "السلام عليكم! أنا HAYO، مساعدك الشخصي الذكي. كيف يمكنني خدمتك؟"

        return "What else can I help you with?"

    def extract_entities(self, message: str) -> dict:
        """Extract important entities from user message."""
        entities = {
            "apps": [],
            "files": [],
            "actions": [],
            "targets": [],
        }

        # Simple extraction patterns
        app_keywords = ["chrome", "edge", "firefox", "excel", "word", "notepad", "vscode", "outlook"]
        for app in app_keywords:
            if app in message.lower():
                entities["apps"].append(app)

        # File patterns
        file_patterns = [r"([a-zA-Z0-9_\-\./\\]+\.\w+)", r"([Dd]esktop|[Dd]ownloads)"]
        import re
        for pattern in file_patterns:
            matches = re.findall(pattern, message)
            entities["files"].extend(matches)

        # Action verbs
        action_verbs = [
            "open", "close", "download", "upload", "create", "delete",
            "convert", "find", "search", "modify", "edit", "send"
        ]
        for verb in action_verbs:
            if verb in message.lower():
                entities["actions"].append(verb)

        return entities

    def format_for_display(
        self,
        response: str,
        execution_details: Optional[str] = None,
    ) -> str:
        """Format response for display to user."""
        formatted = response

        if execution_details:
            formatted += f"\n\n📋 Details:\n{execution_details}"

        return formatted
