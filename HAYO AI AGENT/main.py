"""
CLI entrypoint for HAYO.

Usage:
    python main.py
    python main.py --once "افتح كروم وحمّل ملف PDF من ..."

Async-first: the canonical workflow uses AsyncSqliteSaver, so we drive the
graph via .ainvoke under asyncio.run.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# Load .env BEFORE importing the graph (LLM clients build at import time).
load_dotenv(Path(__file__).parent / ".env", override=False)

from agent.workflow import compile_graph  # noqa: E402

DEFAULT_WORKSPACE = Path(os.getenv("DEFAULT_WORKSPACE", str(Path(__file__).parent)))
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "google").lower().strip()

console = Console()


def assert_keys_present() -> None:
    if MODEL_PROVIDER == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "MODEL_PROVIDER='anthropic' but ANTHROPIC_API_KEY is empty in .env"
        )
    if MODEL_PROVIDER == "google" and not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "MODEL_PROVIDER='google' but GOOGLE_API_KEY is empty in .env"
        )
    if MODEL_PROVIDER == "groq" and not os.getenv("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"):
        raise RuntimeError(
            "MODEL_PROVIDER='groq' but GROQ_API_KEY is empty in .env"
        )


def _print_response(state: dict) -> None:
    msgs = state.get("messages", [])
    for msg in msgs[-3:]:
        kind = msg.__class__.__name__
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if not content.strip():
            continue
        console.print(Panel(Markdown(content[:4000]), title=kind, border_style="cyan"))


async def run_one(graph, message: str, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    initial = {
        "messages": [HumanMessage(content=message)],
        "plan": [],
        "completed_steps": [],
        "workspace": str(DEFAULT_WORKSPACE),
        "error_logs": [],
        "iteration_count": 0,
        "requires_human_approval": False,
        "pending_command": "",
    }
    try:
        final = await graph.ainvoke(initial, config=config)
    except Exception as exc:
        console.print(f"[red]Graph error: {type(exc).__name__}: {exc}[/red]")
        return
    _print_response(final)


async def amain() -> int:
    parser = argparse.ArgumentParser(description="HAYO local OS agent")
    parser.add_argument("--once", type=str, help="Run a single message and exit")
    args = parser.parse_args()

    try:
        assert_keys_present()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    console.print(
        Panel.fit(
            f"[bold green]HAYO AI AGENT[/bold green] — provider: [cyan]{MODEL_PROVIDER}[/cyan]\n"
            f"workspace: {DEFAULT_WORKSPACE}\n"
            f"اكتب طلبك بالعربية أو الإنجليزية. اكتب /quit للخروج.",
            border_style="green",
        )
    )

    graph = compile_graph()
    thread_id = str(uuid.uuid4())

    if args.once:
        await run_one(graph, args.once, thread_id)
        return 0

    while True:
        try:
            user = console.input("[bold yellow]> [/bold yellow]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return 0
        if not user:
            continue
        if user.lower() in ("/quit", "/exit", "exit", "quit"):
            return 0
        await run_one(graph, user, thread_id)


def main() -> int:
    return asyncio.run(amain())


if __name__ == "__main__":
    sys.exit(main())
