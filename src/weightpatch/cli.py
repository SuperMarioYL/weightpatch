"""Command-line entry point.

Commands are wired in as their modules land: ``diff`` today; ``show``, ``apply``
and ``revert`` follow with ``patch.py``, ``screen`` with ``screen.py``.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import __version__
from .checkpoint import Checkpoint, CheckpointError
from .diff import CheckpointDiff, DiffError, ProgressCallback, diff_checkpoints

app = typer.Typer(
    name="weightpatch",
    help="Diff, apply, revert and screen sparse patches for edited checkpoints.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()
error_console = Console(stderr=True)


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"weightpatch {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_show_version,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """A patch file for edited open-weight checkpoints."""


@app.command()
def diff(
    base: Path = typer.Argument(..., help="Base checkpoint directory."),
    target: Path = typer.Argument(..., help="Edited checkpoint directory."),
    as_json: bool = typer.Option(
        False, "--json", help="Print the manifest document instead of a table."
    ),
) -> None:
    """Compare two safetensors checkpoint directories, tensor by tensor."""
    try:
        with Checkpoint(base) as base_checkpoint, Checkpoint(target) as target_checkpoint:
            if as_json:
                result = diff_checkpoints(base_checkpoint, target_checkpoint)
            else:
                with _progress_printer() as progress:
                    result = diff_checkpoints(
                        base_checkpoint, target_checkpoint, progress=progress
                    )
    except (CheckpointError, DiffError, OSError) as exc:
        error_console.print(f"[red]error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc

    if as_json:
        typer.echo(json.dumps(result.manifest(), indent=2, ensure_ascii=False))
        return

    _render(result, base, target)


@contextmanager
def _progress_printer() -> Iterator[ProgressCallback]:
    """Show the tensor being compared on stderr, leaving stdout for the report."""
    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        console=error_console,
        transient=True,
    ) as progress:
        task = progress.add_task("reading checkpoint")

        def report(name: str, index: int, total: int) -> None:
            progress.update(task, description=f"[{index + 1}/{total}] {escape(name)}")

        yield report


def _render(result: CheckpointDiff, base: Path, target: Path) -> None:
    console.print(f"base    {escape(str(base))}")
    console.print(f"target  {escape(str(target))}")
    console.print()

    if not result.tensors:
        console.print("[green]no tensor changes[/green]")
    else:
        table = Table(box=None, header_style="bold", pad_edge=False)
        table.add_column("tensor")
        table.add_column("dtype")
        table.add_column("shape", justify="right")
        table.add_column("ranges")
        table.add_column("before", style="dim")
        table.add_column("after", style="dim")
        for tensor in result.tensors:
            table.add_row(
                tensor.name,
                tensor.dtype,
                _format_shape(tensor.shape),
                _format_ranges(tensor.ranges),
                tensor.before_sha256[:12],
                tensor.after_sha256[:12],
            )
        console.print(table)

    unchanged = result.compared - len(result.tensors)
    console.print(
        f"{len(result.tensors)} changed · {unchanged} unchanged"
        f" · {len(result.added)} added · {len(result.removed)} removed"
    )
    for name in result.added:
        console.print(f"[yellow]added[/yellow]   {escape(name)}")
    for name in result.removed:
        console.print(f"[yellow]removed[/yellow] {escape(name)}")


def _format_shape(shape: tuple[int, ...]) -> str:
    return "×".join(str(dim) for dim in shape) if shape else "scalar"


def _format_ranges(ranges: list[list[int]]) -> str:
    return " ".join(f"[{dim}:{start}:{end}]" for dim, start, end in ranges) or "scalar"


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
