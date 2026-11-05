"""Command-line entry point.

Only the app shell lives here for now: the ``diff``, ``show``, ``apply``,
``revert`` and ``screen`` commands are wired in as their modules land.
"""

from __future__ import annotations

import typer

from . import __version__

app = typer.Typer(
    name="weightpatch",
    help="Diff, apply, revert and screen sparse patches for edited checkpoints.",
    no_args_is_help=True,
    add_completion=False,
)


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


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
