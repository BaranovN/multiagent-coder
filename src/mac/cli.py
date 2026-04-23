"""Command-line interface. `uv run mac solve ...`"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import available_models, load_config
from .orchestrator import run_task
from .state import RunState

app = typer.Typer(add_completion=False, no_args_is_help=True, help="multiagent-coder")
console = Console()


@app.callback()
def _root(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="debug logs")] = False,
) -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _preflight() -> None:
    cfg = load_config()
    avail = available_models(cfg)
    if not avail:
        required = sorted({m.api_key_env for m in cfg.models.values() if m.api_key_env})
        console.print(
            Panel.fit(
                "[red]No LLM providers configured.[/red]\n"
                f"Set at least one of: {', '.join(required)}\n"
                "See .env.example.",
                title="Missing API keys",
            )
        )
        raise typer.Exit(code=2)


def _save_run(state: RunState) -> Path:
    runs_dir = Path(os.getenv("MAC_RUNS_DIR", "runs"))
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = runs_dir / ts
    out.mkdir(parents=True, exist_ok=True)
    (out / "state.json").write_text(state.model_dump_json(indent=2))
    files_dir = out / "code"
    files_dir.mkdir(exist_ok=True)
    for rel, content in state.files.items():
        p = files_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return out


def _render_summary(state: RunState, out_dir: Path) -> None:
    console.rule("[bold]Run summary[/bold]")
    console.print(f"Task: {state.task[:120]}{'…' if len(state.task) > 120 else ''}")
    console.print(f"Language: {state.language}")
    console.print(f"Iterations: {state.iteration}")
    console.print(f"Tokens spent: {state.tokens_spent}")
    if state.review:
        color = "green" if state.review.approved else "red"
        console.print(
            f"Review: [{color}]{'APPROVED' if state.review.approved else 'REJECTED'}[/{color}] — {state.review.verdict}"
        )
    if state.test_report:
        console.print(f"Tests: {state.test_report.summary}")

    t = Table(title="Agent trace", show_lines=False)
    t.add_column("#", justify="right")
    t.add_column("agent")
    t.add_column("model")
    t.add_column("tokens in/out", justify="right")
    t.add_column("ms", justify="right")
    t.add_column("summary")
    for i, h in enumerate(state.history, start=1):
        t.add_row(
            str(i),
            h.agent,
            h.model,
            f"{h.tokens_in}/{h.tokens_out}",
            str(h.duration_ms),
            h.summary[:100],
        )
    console.print(t)
    console.print(f"Artifacts: [cyan]{out_dir}[/cyan]")


@app.command()
def solve(
    task_file: Annotated[
        Path | None,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Markdown/plaintext file describing the task.",
        ),
    ] = None,
    task: Annotated[
        str | None,
        typer.Option("--task", "-t", help="Inline task text."),
    ] = None,
    language: Annotated[
        str | None,
        typer.Option("--language", "-l", help="Force target language (e.g. Rust)."),
    ] = None,
) -> None:
    """Solve a coding task end-to-end."""
    _preflight()
    if not task_file and not task:
        console.print("[red]Provide either TASK_FILE or --task[/red]")
        raise typer.Exit(code=2)
    task_text = task or task_file.read_text()  # type: ignore[union-attr]

    console.print(
        Panel.fit(
            task_text[:800] + ("…" if len(task_text) > 800 else ""),
            title="Task",
        )
    )
    try:
        state = asyncio.run(run_task(task_text, language=language))
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted[/yellow]")
        raise typer.Exit(code=130) from None

    out_dir = _save_run(state)
    _render_summary(state, out_dir)


@app.command()
def doctor() -> None:
    """Check which LLM providers are reachable with current env."""
    load_dotenv()
    cfg = load_config()
    t = Table(title="Model availability")
    t.add_column("key")
    t.add_column("model")
    t.add_column("env var")
    t.add_column("available")
    for key, spec in cfg.models.items():
        ok = spec.api_key_env is None or bool(os.getenv(spec.api_key_env))
        t.add_row(key, spec.model, spec.api_key_env or "-", "yes" if ok else "no")
    console.print(t)

    t2 = Table(title="Agents → model chain")
    t2.add_column("agent")
    t2.add_column("primary")
    t2.add_column("fallback")
    for name, a in cfg.agents.items():
        t2.add_row(name, a.model, ", ".join(a.fallback) or "-")
    console.print(t2)


@app.command()
def config_dump() -> None:
    """Print the resolved config as JSON."""
    cfg = load_config()
    typer.echo(json.dumps(cfg.model_dump(), indent=2))


if __name__ == "__main__":
    app()
