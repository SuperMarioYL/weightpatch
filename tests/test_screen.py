"""m3: screen reports what a derivative artifact kept, lost or repacked."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from weightpatch import Checkpoint, diff_checkpoints, read_patch, screen, write_patch
from weightpatch.cli import app

runner = CliRunner()

Q_PROJ = "model.layers.0.self_attn.q_proj.weight"
UP_PROJ = "model.layers.0.mlp.up_proj.weight"


def test_safetensors_screen_keeps_an_unquantized_edit(checkpoint_pair, patch_file):
    _, edited = checkpoint_pair

    report = screen(edited, read_patch(patch_file))

    assert report.kind == "safetensors"
    assert report.ok
    assert [row.status for row in report.rows] == ["kept", "kept"]
    assert report.summary() == "2 kept · 0 missing · 0 reshaped"


def test_safetensors_screen_reports_the_pre_edit_export(checkpoint_pair, patch_file):
    base, _ = checkpoint_pair

    report = screen(base, read_patch(patch_file))

    assert not report.ok
    assert [row.status for row in report.rows] == ["missing", "missing"]
    assert "still stores the pre-edit tensor" in report.rows[0].detail


def test_safetensors_screen_reports_a_packed_export(packed_export, patch_file):
    report = screen(packed_export, read_patch(patch_file))

    assert not report.ok
    assert [row.status for row in report.rows] == ["reshaped", "reshaped"]
    packed = next(row for row in report.rows if row.name == Q_PROJ)
    assert packed.found_dtype == "U32"
    assert packed.found_shape == (4, 4)
    assert packed.companions == (
        "model.layers.0.self_attn.q_proj.biases",
        "model.layers.0.self_attn.q_proj.scales",
    )


def test_gguf_screen_maps_names_and_reverses_ne_shape(make_gguf, patch_file):
    path = make_gguf({"blk.0.attn_q.weight": (8, 8), "blk.0.ffn_up.weight": (8, 8)})

    report = screen(path, read_patch(patch_file))

    assert report.kind == "gguf"
    assert report.arch == "qwen3"
    assert report.ok
    by_name = {row.name: row for row in report.rows}
    assert by_name[Q_PROJ].mapped_name == "blk.0.attn_q.weight"
    assert by_name[UP_PROJ].mapped_name == "blk.0.ffn_up.weight"
    assert by_name[Q_PROJ].found_shape == (8, 8)
    assert "raw ne 8×8 reverses to the recorded 8×8" in by_name[Q_PROJ].detail


def test_gguf_screen_reports_an_absent_tensor(make_gguf, patch_file):
    path = make_gguf({"blk.0.ffn_up.weight": (8, 8)})

    report = screen(path, read_patch(patch_file))

    by_name = {row.name: row for row in report.rows}
    assert by_name[Q_PROJ].status == "missing"
    assert by_name[Q_PROJ].mapped_name == "blk.0.attn_q.weight"
    assert by_name[UP_PROJ].status == "kept"
    assert not report.ok


def test_gguf_screen_reports_a_reshaped_tensor(make_gguf, patch_file):
    path = make_gguf({"blk.0.attn_q.weight": (8, 9), "blk.0.ffn_up.weight": (8, 8)})

    report = screen(path, read_patch(patch_file))

    by_name = {row.name: row for row in report.rows}
    assert by_name[Q_PROJ].status == "reshaped"
    assert by_name[Q_PROJ].found_shape == (8, 9)
    assert not report.ok


def test_gguf_screen_reports_an_unsupported_architecture(make_gguf, patch_file):
    path = make_gguf(
        {"blk.0.attn_q.weight": (8, 8)}, arch="acme", name="unsupported"
    )

    report = screen(path, read_patch(patch_file))

    assert report.arch == "acme"
    assert [row.status for row in report.rows] == ["unsupported-arch", "unsupported-arch"]
    assert not report.ok


def test_gguf_screen_collapses_an_unmapped_namespace(make_gguf, linear_attention_patch):
    path = make_gguf(
        {"blk.0.attn_q.weight": (8, 8)},
        arch="qwen35",
        block_count=4,
        name="qwen35",
    )

    report = screen(path, linear_attention_patch)

    assert len(report.rows) == 1
    assert report.rows[0].status == "unmapped-arch"
    assert "resolved none" in report.rows[0].detail
    assert not report.ok


def test_cli_screen_exit_codes(checkpoint_pair, packed_export, patch_file):
    base, edited = checkpoint_pair

    kept = runner.invoke(app, ["screen", str(edited), str(patch_file)])
    assert kept.exit_code == 0, kept.output
    assert "2 kept · 0 missing · 0 reshaped" in kept.stdout

    lost = runner.invoke(app, ["screen", str(packed_export), str(patch_file)])
    assert lost.exit_code == 1
    assert "reshaped" in lost.stdout

    missing = runner.invoke(app, ["screen", str(base), str(patch_file), "--json"])
    assert missing.exit_code == 1
    document = json.loads(missing.stdout)
    assert document["ok"] is False
    assert document["counts"]["missing"] == 2


def test_cli_screen_on_a_gguf_exits_zero(make_gguf, patch_file):
    path = make_gguf({"blk.0.attn_q.weight": (8, 8), "blk.0.ffn_up.weight": (8, 8)})

    result = runner.invoke(app, ["screen", str(path), str(patch_file)])

    assert result.exit_code == 0, result.output
    assert "2 kept · 0 missing · 0 reshaped" in result.stdout
    assert "arch qwen3" in result.stdout


def test_cli_screen_exits_non_zero_when_nothing_was_screened(make_checkpoint, tmp_path):
    base = make_checkpoint("base")
    edited = make_checkpoint(
        "edited", extra={"model.layers.0.rotary_emb.inv_freq": ((8,), 21)}
    )
    with Checkpoint(base) as a, Checkpoint(edited) as b:
        patch = write_patch(diff_checkpoints(a, b), tmp_path / "added.wpatch")

    result = runner.invoke(app, ["screen", str(base), str(patch.path)])

    assert result.exit_code == 1
    assert "nothing screened" in result.output
