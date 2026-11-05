"""Command-line entry point for ``weightpatch``.

One process, no server. Every command prints a human report on stdout and
accepts ``--json`` for the manifest or report document; errors go to stderr
with exit code 1. ``screen`` exits non-zero when a declared edit did not
survive, so it can gate a release pipeline.
"""

from __future__ import annotations

import json
import textwrap
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import __version__
from .checkpoint import Checkpoint, CheckpointError
from .diff import CheckpointDiff, DiffError, ProgressCallback, TensorDiff, diff_checkpoints
from .patch import (
    Patch,
    PatchError,
    TensorEdit,
    apply_patch,
    patch_id,
    read_patch,
    revert_patch,
    write_patch,
)
from .screen import ScreenError, ScreenReport
from .screen import screen as screen_artifact

app = typer.Typer(
    name="weightpatch",
    help="Diff, apply, revert, stack and screen sparse patches for edited checkpoints.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console(highlight=False)
error_console = Console(stderr=True, highlight=False)

_CLI_ERRORS = (CheckpointError, DiffError, PatchError, ScreenError, OSError)
_PATCH_HELP = "A .wpatch file written by `weightpatch diff`."
_STATUS_STYLE = {
    "kept": "green",
    "missing": "red",
    "reshaped": "yellow",
    "unsupported-arch": "red",
    "unmapped-arch": "yellow",
}


def _fail(exc: Exception) -> typer.Exit:
    error_console.print(f"[red]error:[/red] {escape(str(exc))}")
    return typer.Exit(code=1)


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


# -- diff ------------------------------------------------------------------


@app.command()
def diff(
    base: Path = typer.Argument(..., help="Base checkpoint directory."),
    target: Path = typer.Argument(..., help="Edited checkpoint directory."),
    output: Path | None = typer.Option(
        None, "-o", "--output", help="Write the patch here instead of only reporting it."
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Print the patch manifest instead of the report."
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
        if output is not None:
            manifest = write_patch(result, output).manifest
        else:
            document = result.manifest()
            manifest = {
                "id": patch_id(
                    document["base"], document["parents"], document["edit"]
                ),
                **document,
            }
    except _CLI_ERRORS as exc:
        raise _fail(exc) from exc

    if as_json:
        typer.echo(json.dumps(manifest, indent=2, ensure_ascii=False))
        return

    _render_diff(result, base, target, output, manifest["id"])


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


def _render_diff(
    result: CheckpointDiff,
    base: Path,
    target: Path,
    output: Path | None,
    content_id: str,
) -> None:
    console.print(f"base    {escape(str(base))}")
    console.print(f"target  {escape(str(target))}")
    console.print()

    if not result.tensors:
        console.print("[green]no tensor changes[/green]")
    else:
        console.print(_edit_table(result.tensors))

    unchanged = result.compared - len(result.tensors)
    console.print(
        f"{len(result.tensors)} changed · {unchanged} unchanged"
        f" · {len(result.added)} added · {len(result.removed)} removed"
    )
    for name in result.added:
        console.print(f"[yellow]added[/yellow]   {escape(name)}")
    for name in result.removed:
        console.print(f"[yellow]removed[/yellow] {escape(name)}")

    if output is not None:
        console.print(f"patch   {escape(str(output))} · {content_id[:12]}")


# -- show ------------------------------------------------------------------


@app.command()
def show(
    patch_file: Path = typer.Argument(..., help=_PATCH_HELP),
    as_json: bool = typer.Option(
        False, "--json", help="Print the manifest instead of the report."
    ),
) -> None:
    """Print the edits recorded in a patch: slices, dtypes and digests."""
    try:
        patch = read_patch(patch_file)
    except _CLI_ERRORS as exc:
        raise _fail(exc) from exc

    if as_json:
        typer.echo(json.dumps(patch.manifest, indent=2, ensure_ascii=False))
        return

    _render_show(patch)


def _render_show(patch: Patch) -> None:
    files = len(patch.base.get("files", []))
    console.print(f"patch   {escape(str(patch.path))}")
    console.print(f"id      {patch.id[:16]}")
    console.print(
        f"base    {files} file(s) · {patch.base.get('tensor_count', '?')} tensor(s)"
    )
    console.print(
        "parents " + (", ".join(parent[:12] for parent in patch.parents) or "-")
    )
    probe = patch.probe
    claim = probe.get("claim") or "(no claim recorded)"
    console.print(f"probe   {escape(str(probe.get('kind', 'declared')))} · {escape(claim)}")
    console.print()

    if patch.edits:
        console.print(_edit_table(patch.edits))
    else:
        console.print("no slice edits recorded")

    console.print(
        f"{len(patch.edits)} tensor edit(s) · "
        f"{len(patch.added)} added · {len(patch.removed)} removed"
    )
    for name in patch.added:
        console.print(f"[yellow]added[/yellow]   {escape(name)}")
    for name in patch.removed:
        console.print(f"[yellow]removed[/yellow] {escape(name)}")


# -- apply, revert, stack --------------------------------------------------


@app.command()
def apply(
    target: Path = typer.Argument(..., help="Checkpoint directory to patch."),
    patch_file: Path = typer.Argument(..., help=_PATCH_HELP),
    output: Path = typer.Option(
        ..., "-o", "--output", help="Write the patched checkpoint here."
    ),
) -> None:
    """Write a copy of TARGET with one patch applied."""
    try:
        patch = read_patch(patch_file)
        apply_patch(target, [patch], output)
    except _CLI_ERRORS as exc:
        raise _fail(exc) from exc

    console.print(
        f"applied {escape(patch.path.name)} · {patch.id[:12]} → {escape(str(output))}"
    )
    console.print(f"{len(patch.edits)} tensor(s) rewritten · post-image verified")


@app.command()
def revert(
    target: Path = typer.Argument(..., help="Checkpoint directory to un-patch."),
    patch_file: Path = typer.Argument(..., help=_PATCH_HELP),
    output: Path = typer.Option(
        ..., "-o", "--output", help="Write the reverted checkpoint here."
    ),
) -> None:
    """Write a copy of TARGET with one patch undone."""
    try:
        patch = read_patch(patch_file)
        revert_patch(target, patch, output)
    except _CLI_ERRORS as exc:
        raise _fail(exc) from exc

    console.print(
        f"reverted {escape(patch.path.name)} · {patch.id[:12]} → {escape(str(output))}"
    )
    console.print(f"{len(patch.edits)} tensor(s) restored · post-image verified")


@app.command()
def stack(
    target: Path = typer.Argument(..., help="Checkpoint directory the chain starts from."),
    patches: list[Path] = typer.Argument(
        ..., help="Two or more patches, in the order they were recorded."
    ),
    output: Path = typer.Option(
        ..., "-o", "--output", help="Write the stacked checkpoint here."
    ),
) -> None:
    """Apply an ordered chain of patches, each verified against the previous state."""
    if len(patches) < 2:
        raise typer.BadParameter("stack needs at least two patches; use apply for one")

    try:
        chain = [read_patch(path) for path in patches]
        apply_patch(target, chain, output)
    except _CLI_ERRORS as exc:
        raise _fail(exc) from exc

    console.print(
        f"stacked {len(chain)} patches onto {escape(str(target))} → {escape(str(output))}"
    )
    for index, patch in enumerate(chain, start=1):
        parents = ", ".join(parent[:12] for parent in patch.parents) or "-"
        console.print(
            f"  {index}. {patch.id[:12]}  {escape(patch.path.name)}  parents: {parents}"
        )
    edits = sum(len(patch.edits) for patch in chain)
    console.print(f"{edits} tensor edit(s) · pre-image chain verified")


# -- screen ----------------------------------------------------------------


@app.command(name="screen")
def screen_command(
    source: Path = typer.Argument(
        ..., help="A .gguf file or a safetensors checkpoint directory."
    ),
    patch_file: Path = typer.Argument(..., help=_PATCH_HELP),
    as_json: bool = typer.Option(
        False, "--json", help="Print the report as JSON instead of a table."
    ),
) -> None:
    """Check whether a patch's edits survived a derivative artifact.

    Exits 0 only when every declared edit is found; 1 when any is missing or
    reshaped, and 1 when there is nothing to screen.
    """
    try:
        patch = read_patch(patch_file)
        report = screen_artifact(source, patch)
    except _CLI_ERRORS as exc:
        raise _fail(exc) from exc

    if as_json:
        typer.echo(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    else:
        _render_screen(report)

    if not report.rows:
        error_console.print(
            "[yellow]nothing screened:[/yellow] the patch declares no slice edits"
        )
        raise typer.Exit(code=1)
    if not report.ok:
        raise typer.Exit(code=1)


def _render_screen(report: ScreenReport) -> None:
    if report.kind == "gguf":
        header = f"{report.source.name} · gguf"
        if report.arch:
            header += f" · arch {report.arch}"
        console.print(escape(header))
    else:
        console.print(f"{escape(report.source.name)} · safetensors")
    console.print()

    for row in report.rows:
        style = _STATUS_STYLE.get(row.status)
        if style:
            console.print(f"{row.status:<16} {escape(row.name)}", style=style)
        else:
            console.print(f"{row.status:<16} {escape(row.name)}")
        _print_indented(row.detail, 17)

    console.print()
    console.print(report.summary() if report.rows else "no rows")
    for note in report.notes:
        console.print(f"[yellow]note[/yellow] {escape(note)}")


# -- helpers ---------------------------------------------------------------


def _print_indented(text: str, indent: int) -> None:
    """Print wrapped detail under a row without trailing fill."""
    width = max(24, console.width - indent)
    lines = textwrap.wrap(
        text, width=width, break_long_words=False, break_on_hyphens=False
    ) or [""]
    for line in lines:
        console.print(f"{'':{indent}}{escape(line)}", style="dim", soft_wrap=True)


def _edit_table(edits: Sequence[TensorDiff | TensorEdit]) -> Table:
    """One row per recorded tensor edit: shape, slices, both digests."""
    table = Table(box=None, header_style="bold", pad_edge=False, show_edge=False)
    table.add_column("tensor", overflow="fold")
    table.add_column("dtype")
    table.add_column("shape", justify="right")
    table.add_column("slice")
    table.add_column("before", style="dim")
    table.add_column("after", style="dim")
    for edit in edits:
        table.add_row(
            edit.name,
            edit.dtype,
            _format_shape(edit.shape),
            _format_ranges(edit.ranges),
            edit.before_sha256[:8],
            edit.after_sha256[:8],
        )
    return table


def _format_shape(shape: tuple[int, ...]) -> str:
    return "×".join(str(dim) for dim in shape) if shape else "scalar"


def _format_ranges(ranges: list[list[int]] | tuple[tuple[int, int, int], ...]) -> str:
    return " ".join(f"[{dim}:{start}:{end}]" for dim, start, end in ranges) or "scalar"


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
