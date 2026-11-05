"""m2: a .wpatch applies, stacks and reverts with pre-image checks."""

from __future__ import annotations

import json
import zipfile

import pytest
from typer.testing import CliRunner

from weightpatch import (
    Checkpoint,
    PatchError,
    apply_patch,
    diff_checkpoints,
    read_patch,
    revert_patch,
    write_patch,
)
from weightpatch.cli import app
from conftest import EDITS, tree

runner = CliRunner()

Q_PROJ = "model.layers.0.self_attn.q_proj.weight"


def test_apply_rebuilds_the_edited_tensors(checkpoint_pair, patch_file, tmp_path):
    base, edited = checkpoint_pair
    output = tmp_path / "applied"

    apply_patch(base, [read_patch(patch_file)], output)

    assert tree(output) == tree(edited)


def test_revert_restores_the_base(checkpoint_pair, patch_file, tmp_path):
    base, edited = checkpoint_pair
    applied = tmp_path / "applied"
    reverted = tmp_path / "reverted"

    patch = read_patch(patch_file)
    apply_patch(base, [patch], applied)
    revert_patch(applied, patch, reverted)

    assert tree(reverted) == tree(base)


def test_apply_refuses_a_moved_base(make_checkpoint, patch_file, tmp_path):
    moved = make_checkpoint("moved", edits={Q_PROJ: {(7, 7): 0xAAAA}})
    output = tmp_path / "applied"

    with pytest.raises(PatchError, match="pre-image digest"):
        apply_patch(moved, [read_patch(patch_file)], output)

    assert not output.exists()


def test_apply_refuses_overlapping_stacked_patches(
    checkpoint_pair, patch_file, make_checkpoint, tmp_path
):
    base, _ = checkpoint_pair
    other = make_checkpoint("other", edits={Q_PROJ: {(1, 3): 0x9999}})
    with Checkpoint(base) as base_checkpoint, Checkpoint(other) as other_checkpoint:
        second = write_patch(
            diff_checkpoints(base_checkpoint, other_checkpoint), tmp_path / "other.wpatch"
        )

    with pytest.raises(PatchError, match="overlapping slices"):
        apply_patch(base, [read_patch(patch_file), second], tmp_path / "applied")


def test_stack_reverts_in_reverse_order(stack_setup, tmp_path):
    stacked = tmp_path / "stacked"
    middle = tmp_path / "middle"
    back = tmp_path / "back"

    apply_patch(stack_setup.base, [stack_setup.first, stack_setup.second], stacked)
    assert tree(stacked) == tree(stack_setup.after_second)

    revert_patch(stacked, stack_setup.second, middle)
    assert tree(middle) == tree(stack_setup.after_first)

    revert_patch(middle, stack_setup.first, back)
    assert tree(back) == tree(stack_setup.base)


def test_read_patch_rejects_a_tampered_manifest(patch_file, tmp_path):
    with zipfile.ZipFile(patch_file) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        values = archive.read("values.npz")
    manifest["edit"]["tensors"][0]["before_sha256"] = "0" * 64

    tampered = tmp_path / "tampered.wpatch"
    with zipfile.ZipFile(tampered, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("values.npz", values)

    with pytest.raises(PatchError, match="changed after it was written"):
        read_patch(tampered)


def test_apply_refuses_a_patch_with_only_added_tensors(make_checkpoint, tmp_path):
    base = make_checkpoint("base")
    edited = make_checkpoint(
        "edited", extra={"model.layers.0.rotary_emb.inv_freq": ((8,), 21)}
    )
    with Checkpoint(base) as base_checkpoint, Checkpoint(edited) as edited_checkpoint:
        patch = write_patch(
            diff_checkpoints(base_checkpoint, edited_checkpoint),
            tmp_path / "added.wpatch",
        )

    assert patch.added == ("model.layers.0.rotary_emb.inv_freq",)
    with pytest.raises(PatchError, match="slice edits only"):
        apply_patch(base, [patch], tmp_path / "applied")


def test_cli_apply_then_revert_restores_the_base(checkpoint_pair, patch_file, tmp_path):
    base, edited = checkpoint_pair
    rebuilt = tmp_path / "rebuilt"
    back = tmp_path / "back"

    applied = runner.invoke(
        app, ["apply", str(base), str(patch_file), "-o", str(rebuilt)]
    )
    assert applied.exit_code == 0, applied.output
    assert tree(rebuilt) == tree(edited)
    assert "post-image verified" in applied.stdout

    reverted = runner.invoke(app, ["revert", str(rebuilt), str(patch_file), "-o", str(back)])
    assert reverted.exit_code == 0, reverted.output
    assert tree(back) == tree(base)


def test_cli_stack_applies_the_chain(stack_setup, tmp_path):
    output = tmp_path / "stacked"

    result = runner.invoke(
        app,
        [
            "stack",
            str(stack_setup.base),
            str(stack_setup.first.path),
            str(stack_setup.second.path),
            "-o",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert tree(output) == tree(stack_setup.after_second)
    assert "pre-image chain verified" in result.stdout
    assert "parents:" in result.stdout


def test_cli_stack_needs_two_patches(stack_setup, tmp_path):
    result = runner.invoke(
        app,
        ["stack", str(stack_setup.base), str(stack_setup.first.path), "-o", str(tmp_path / "x")],
    )
    assert result.exit_code == 2
    assert "at least two patches" in result.output


def test_cli_show_prints_the_edit_table(patch_file):
    result = runner.invoke(app, ["show", str(patch_file)])
    assert result.exit_code == 0, result.output
    assert "2 tensor edit(s)" in result.stdout
    assert "[0:1:3]" in result.stdout
    assert "BF16" in result.stdout


def test_cli_show_json_round_trips_the_manifest(patch_file):
    result = runner.invoke(app, ["show", str(patch_file), "--json"])
    assert result.exit_code == 0, result.output
    document = json.loads(result.stdout)
    assert len(document["id"]) == 64
    assert len(document["edit"]["tensors"]) == 2
    assert sorted(EDITS) == sorted(entry["name"] for entry in document["edit"]["tensors"])
