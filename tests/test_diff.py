"""m1: a checkpoint pair diffs into a sparse, content-addressed patch."""

from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from weightpatch import Checkpoint, DiffError, diff_checkpoints, read_patch
from weightpatch.cli import app
from conftest import DEFAULT_TENSORS

runner = CliRunner()

UP_PROJ = "model.layers.0.mlp.up_proj.weight"
Q_PROJ = "model.layers.0.self_attn.q_proj.weight"


def by_name_ranges(result, name: str) -> list[list[int]]:
    return next(t.ranges for t in result.tensors if t.name == name)


def test_diff_keeps_only_changed_tensors(checkpoint_pair):
    base, edited = checkpoint_pair
    with Checkpoint(base) as base_checkpoint, Checkpoint(edited) as edited_checkpoint:
        result = diff_checkpoints(base_checkpoint, edited_checkpoint)

    assert [tensor.name for tensor in result.tensors] == [UP_PROJ, Q_PROJ]
    assert result.compared == len(DEFAULT_TENSORS)
    assert result.added == [] and result.removed == []


def test_diff_records_the_changed_bounding_box(checkpoint_pair):
    base, edited = checkpoint_pair
    with Checkpoint(base) as base_checkpoint, Checkpoint(edited) as edited_checkpoint:
        result = diff_checkpoints(base_checkpoint, edited_checkpoint)

    by_name = {tensor.name: tensor for tensor in result.tensors}
    assert by_name[Q_PROJ].ranges == [[0, 1, 3], [1, 2, 6]]
    assert by_name[UP_PROJ].ranges == [[0, 0, 1], [1, 1, 7]]
    assert by_name[Q_PROJ].shape == (8, 8)
    assert by_name[Q_PROJ].dtype == "BF16"
    assert by_name[Q_PROJ].element_size == 2
    assert by_name[Q_PROJ].before_sha256 != by_name[Q_PROJ].after_sha256


def test_diff_stores_raw_bf16_words(checkpoint_pair):
    base, edited = checkpoint_pair
    with (
        Checkpoint(base) as base_checkpoint,
        Checkpoint(edited) as edited_checkpoint,
    ):
        result = diff_checkpoints(base_checkpoint, edited_checkpoint)
        edited_slab = edited_checkpoint.read_slab(Q_PROJ, by_name_ranges(result, Q_PROJ))

    tensor = next(t for t in result.tensors if t.name == Q_PROJ)
    assert tensor.before.dtype == np.uint16
    assert tensor.after.dtype == np.uint16
    assert tensor.before.shape == (2, 4)
    assert np.array_equal(tensor.after, edited_slab)
    # The two changed elements are at local (0, 0) and (1, 3).
    assert int(tensor.after[0, 0]) == 0x1000
    assert int(tensor.after[1, 3]) == 0x2000


def test_diff_digests_match_the_checkpoint(checkpoint_pair):
    base, edited = checkpoint_pair
    with (
        Checkpoint(base) as base_checkpoint,
        Checkpoint(edited) as edited_checkpoint,
    ):
        result = diff_checkpoints(base_checkpoint, edited_checkpoint)
        before = base_checkpoint.tensor_sha256(Q_PROJ)
        after = edited_checkpoint.tensor_sha256(Q_PROJ)

    tensor = next(t for t in result.tensors if t.name == Q_PROJ)
    assert (tensor.before_sha256, tensor.after_sha256) == (before, after)


def test_diff_of_a_checkpoint_with_itself_is_empty(checkpoint_pair):
    base, _ = checkpoint_pair
    with Checkpoint(base) as base_checkpoint:
        result = diff_checkpoints(base_checkpoint, base_checkpoint)
    assert result.tensors == []
    assert result.compared == len(DEFAULT_TENSORS)


def test_diff_reports_added_and_removed_names(make_checkpoint):
    base = make_checkpoint("base")
    edited = make_checkpoint(
        "edited",
        drop=["model.norm.weight"],
        extra={"model.layers.0.rotary_emb.inv_freq": ((8,), 21)},
    )
    with Checkpoint(base) as base_checkpoint, Checkpoint(edited) as edited_checkpoint:
        result = diff_checkpoints(base_checkpoint, edited_checkpoint)

    assert result.removed == ["model.norm.weight"]
    assert result.added == ["model.layers.0.rotary_emb.inv_freq"]
    assert result.tensors == []


def test_diff_refuses_a_reshaped_tensor(make_checkpoint):
    layout = dict(DEFAULT_TENSORS)
    layout[Q_PROJ] = ((8, 9), 5)
    base = make_checkpoint("base")
    edited = make_checkpoint("edited", tensors=layout)

    with Checkpoint(base) as base_checkpoint, Checkpoint(edited) as edited_checkpoint:
        with pytest.raises(DiffError, match="slice-comparable"):
            diff_checkpoints(base_checkpoint, edited_checkpoint)


def test_cli_diff_writes_a_patch_and_reports_it(checkpoint_pair, tmp_path):
    base, edited = checkpoint_pair
    output = tmp_path / "cli.wpatch"

    result = runner.invoke(app, ["diff", str(base), str(edited), "-o", str(output)])

    assert result.exit_code == 0, result.output
    assert output.is_file()
    patch = read_patch(output)
    assert len(patch.edits) == 2
    assert "2 changed · 2 unchanged" in result.stdout
    assert "patch" in result.stdout


def test_cli_diff_json_is_stable(checkpoint_pair):
    base, edited = checkpoint_pair

    first = runner.invoke(app, ["diff", str(base), str(edited), "--json"])
    second = runner.invoke(app, ["diff", str(base), str(edited), "--json"])

    assert first.exit_code == 0, first.output
    assert first.stdout == second.stdout
    document = json.loads(first.stdout)
    assert len(document["id"]) == 64
    assert document["parents"] == []
    assert [entry["name"] for entry in document["edit"]["tensors"]] == [UP_PROJ, Q_PROJ]
    assert document["edit"]["tensors"][1]["ranges"] == [[0, 1, 3], [1, 2, 6]]


def test_cli_diff_reports_a_missing_directory(checkpoint_pair, tmp_path):
    _, edited = checkpoint_pair
    result = runner.invoke(app, ["diff", str(tmp_path / "absent"), str(edited)])
    assert result.exit_code == 1
    assert "error:" in result.output


def test_edit_values_are_the_base_slab(checkpoint_pair):
    base, edited = checkpoint_pair
    with Checkpoint(base) as base_checkpoint, Checkpoint(edited) as edited_checkpoint:
        result = diff_checkpoints(base_checkpoint, edited_checkpoint)
        base_slab = base_checkpoint.read_slab(Q_PROJ, by_name_ranges(result, Q_PROJ))

    tensor = next(t for t in result.tensors if t.name == Q_PROJ)
    assert np.array_equal(tensor.before, base_slab)
    assert int(tensor.before[0, 0]) != int(tensor.after[0, 0])
