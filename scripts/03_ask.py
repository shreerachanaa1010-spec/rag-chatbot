"""
Step 8 runner: interactive CLI for the HR Policy & Benefits Resolution Engine.

Usage:
    python scripts/03_ask.py
"""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from hr_rag.pipeline import ask

console = Console()


def main() -> None:
    console.print(Panel.fit(
        "HR Policy & Benefits Resolution Engine\n"
        "Ask a question as an employee would. Type 'exit' to quit.",
        title="hr-rag",
    ))

    while True:
        question = Prompt.ask("\n[bold cyan]Employee question[/bold cyan]")
        if question.strip().lower() in {"exit", "quit"}:
            break

        region = Prompt.ask(
            "[bold cyan]Employee region[/bold cyan] (US / EU / blank for none)",
            default="",
        ).strip() or None

        console.print("[dim]Retrieving policy excerpts and generating response...[/dim]")
        result = ask(question, region=region)

        console.print(Panel(
            f"[bold]Answer:[/bold] {result.decision.answer}\n\n"
            f"[bold]Confidence:[/bold] {result.decision.confidence}\n"
            f"[bold]Conflict flag:[/bold] {result.decision.conflict_flag}\n"
            f"[bold]Next action:[/bold] {result.decision.next_action}\n\n"
            f"[bold]Cited sections:[/bold]\n" + "\n".join(f"  - {c}" for c in result.decision.cited_sections),
            title="Structured Decision",
            border_style="green" if not result.decision.conflict_flag else "red",
        ))

        console.print(Panel(result.draft_email, title="Drafted Response Email", border_style="blue"))

        console.print("[dim]Retrieved chunks used (for debugging/transparency):[/dim]")
        for rc in result.retrieved:
            console.print(
                f"  [dim]- {rc.chunk.doc_id} {rc.chunk.version} "
                f"(chunk {rc.chunk.chunk_index}, distance={rc.distance:.3f})[/dim]"
            )


if __name__ == "__main__":
    main()
