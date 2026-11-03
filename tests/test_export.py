"""Tests for the ONNX stage of the deployment path.

The point of these is that an operator or graph change that would only fail
once a board is in the loop should fail here instead. They do not need
ESP-PPQ (which pins python <3.13) or hardware.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "training"))
sys.path.insert(0, str(_REPO / "export"))

from models import TILE_SIZE, TileStudent  # noqa: E402
from operators import ESPDL_INT8_OPS, RECOMMENDED_OPSET, unsupported_ops  # noqa: E402
from to_espdl import check_conv_groups, export_onnx, strip_identity  # noqa: E402


@pytest.mark.parametrize("in_channels", [1, 2])
def test_export_graph_is_deployable(tmp_path, in_channels):
    model = TileStudent(in_channels=in_channels)
    report = export_onnx(model, tmp_path / "m.onnx", in_channels)
    assert report["unsupported"] == []
    assert report["conv_group_problems"] == []
    assert report["opset"] == RECOMMENDED_OPSET
    assert set(report["op_types"]) <= ESPDL_INT8_OPS


def test_batchnorm_is_folded_away(tmp_path):
    """ESP-DL has no BatchNormalization kernel, so none may survive export."""
    report = export_onnx(TileStudent(in_channels=1), tmp_path / "m.onnx", 1)
    assert not any("BatchNorm" in op for op in report["op_types"])


def test_untrained_export_emits_identity_and_we_strip_it(tmp_path):
    """Regression test for the artefact that would look like a blocker.

    An untrained model has identical zero folded biases, ONNX deduplicates
    them, and the duplicates become Identity aliases -- which ESP-DL does not
    implement. Exporting must remove them rather than fail.
    """
    assert "Identity" not in ESPDL_INT8_OPS
    report = export_onnx(TileStudent(in_channels=1), tmp_path / "m.onnx", 1)
    assert report["identity_removed"] > 0, "expected the untrained artefact"
    assert "Identity" not in report["op_types"]


def test_trained_model_needs_no_stripping(tmp_path):
    """With distinct BatchNorm statistics the artefact does not arise."""
    model = TileStudent(in_channels=1)
    gen = torch.Generator().manual_seed(0)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean.normal_(generator=gen)
            module.running_var.uniform_(0.5, 2.0, generator=gen)
            module.bias.data.normal_(generator=gen)
    report = export_onnx(model, tmp_path / "m.onnx", 1)
    assert report["identity_removed"] == 0


@pytest.mark.parametrize("in_channels", [1, 2])
def test_stripping_identity_preserves_numerics(tmp_path, in_channels):
    """The rewiring must not change what the graph computes."""
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(0)
    model = TileStudent(in_channels=in_channels).eval()
    path = tmp_path / "m.onnx"
    export_onnx(model, path, in_channels)

    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, (1, in_channels, TILE_SIZE, TILE_SIZE)).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    got = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    ).run(None, {"tile": x})[0]
    assert np.allclose(expected, got, atol=1e-5)


def test_depthwise_convolutions_pass_the_group_restriction(tmp_path):
    """ESP-DL allows groups of 1 or input_channels only; ours are depthwise."""
    export_onnx(TileStudent(in_channels=2), tmp_path / "m.onnx", 2)
    graph = onnx.load(str(tmp_path / "m.onnx"))
    assert check_conv_groups(graph) == []
    groups = [
        next((a.i for a in n.attribute if a.name == "group"), 1)
        for n in graph.graph.node if n.op_type == "Conv"
    ]
    assert sorted(set(groups)) == [1, 16, 32, 64, 96, 128]


def test_illegal_conv_groups_are_reported(tmp_path):
    """A grouped-but-not-depthwise conv must be caught before it reaches a board."""
    class Grouped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(8, 8, 3, groups=4, bias=False)

        def forward(self, x):
            return self.conv(x)

    path = tmp_path / "g.onnx"
    torch.onnx.export(Grouped().eval(), torch.zeros(1, 8, 16, 16), str(path),
                      opset_version=RECOMMENDED_OPSET, dynamo=False)
    assert check_conv_groups(onnx.load(str(path))) != []


def test_unsupported_ops_helper():
    assert unsupported_ops(["Conv", "Relu", "Gemm"]) == []
    assert unsupported_ops(["Conv", "Identity", "Einsum"]) == ["Einsum", "Identity"]


def test_strip_identity_returns_zero_on_a_clean_graph(tmp_path):
    path = tmp_path / "m.onnx"
    export_onnx(TileStudent(in_channels=1), path, 1)
    assert strip_identity(onnx.load(str(path))) == 0


@pytest.mark.parametrize("exporter", ["legacy", "dynamo"])
def test_export_is_a_single_self_contained_file(tmp_path, exporter):
    """No .onnx.data sidecar may be required.

    The dynamo exporter spills most initialisers to external data by default,
    which would mean copying the .onnx to Colab silently loses the weights.
    """
    if exporter == "dynamo":
        pytest.importorskip("onnxscript")
    from to_espdl import external_initializers

    path = tmp_path / "m.onnx"
    export_onnx(TileStudent(in_channels=1), path, 1, exporter=exporter)
    assert not list(tmp_path.glob("*.onnx.data")), "left an external-data sidecar"
    assert external_initializers(onnx.load(str(path), load_external_data=False)) == []
    # Weights really are inside: a 61K-parameter float32 model is ~240 KB.
    assert path.stat().st_size > 200_000


@pytest.mark.parametrize("exporter", ["legacy", "dynamo"])
def test_both_exporters_produce_deployable_graphs(tmp_path, exporter):
    if exporter == "dynamo":
        pytest.importorskip("onnxscript")
    report = export_onnx(TileStudent(in_channels=1), tmp_path / "m.onnx", 1,
                         exporter=exporter)
    assert report["unsupported"] == []
    assert set(report["op_types"]) <= ESPDL_INT8_OPS
    assert report["exporter"] == exporter
