"""ESP-DL operator support, vendored so the export path can check itself.

The authoritative list lives in ``operator_support_state.md`` in the ESP-DL
repository. It is copied here because the export script has to be able to say
"this graph will not run on the chip" *before* anyone installs ESP-PPQ or
touches a board, and because pinning it makes an operator regression visible
when the list is refreshed.

Provenance
----------
Source : https://github.com/espressif/esp-dl/blob/master/operator_support_state.md
Commit : 5989f4a
Dated  : 2026-11-02
Checked: 2026-11-03

ESP-DL states that its operator interface is aligned with ONNX and recommends
**opset 18** for export. The 68 operators below are the ones marked as
implemented for ``int8`` (the ``w8a8`` scheme this project deploys). Operators
supported only in ``int16``/``float32`` are deliberately excluded: a graph that
needs one of those cannot be deployed as w8a8.

Refresh this list by re-reading the file above and noting the new commit.
"""

from __future__ import annotations

#: Operators ESP-DL implements for int8 (w8a8) inference. See module docstring.
ESPDL_INT8_OPS: frozenset[str] = frozenset({
    "Add", "Attention", "AveragePool", "Clip", "Concat", "Conv", "ConvTranspose",
    "DepthToSpace", "Div", "Elu", "Equal", "Exp", "Flatten", "GRU", "Gather",
    "Gemm", "GlobalAveragePool", "Greater", "GreaterOrEqual", "HardSigmoid",
    "HardSwish", "InverseSigmoid", "LSTM", "LayerNormalization", "LeakyRelu",
    "Less", "LessOrEqual", "Log", "LogSoftmax", "LpNormalization", "MatMul",
    "MaxPool", "Mod", "Mul", "Neg", "PRelu", "Pad", "Pow", "RMSNormalization",
    "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceMax",
    "ReduceMean", "ReduceMin", "ReduceProd", "ReduceSum", "ReduceSumSquare",
    "Relu", "Reshape", "Resize", "ReverseSequence", "RotaryEmbedding",
    "ScatterND", "Sigmoid", "Slice", "Softmax", "SpaceToDepth", "Split",
    "Sqrt", "Squeeze", "Sub", "Swish", "Tanh", "TopK", "Transpose", "Unsqueeze",
})

#: ESP-DL's recommended ONNX opset.
RECOMMENDED_OPSET = 18

#: Operators ESP-DL does *not* implement but which a PyTorch export can emit
#: anyway, mapped to what to do about it. Identity in particular shows up only
#: when exporting an *untrained* model: with freshly initialised BatchNorms
#: every folded convolution bias is the same zero vector, ONNX constant-folding
#: deduplicates them, and the duplicates become Identity aliases. A trained
#: model exports a clean graph. ``strip_identity`` removes them either way.
KNOWN_EXPORT_ARTEFACTS: dict[str, str] = {
    "Identity": (
        "constant-aliasing artefact of exporting an untrained model; "
        "removed by export.to_espdl.strip_identity()"
    ),
}

#: Restrictions worth asserting rather than discovering on the chip.
#: ESP-DL's Conv supports ``groups`` of either 1 or ``input_channels`` only --
#: i.e. dense and fully depthwise convolutions, but nothing in between.
CONV_ALLOWED_GROUPS = "1 or input_channels (dense or fully depthwise)"


def unsupported_ops(op_types: object) -> list[str]:
    """Operator types in ``op_types`` that ESP-DL cannot run as int8.

    Args:
        op_types: any iterable of ONNX operator type names.

    Returns:
        Sorted, de-duplicated list of unsupported names.
    """
    return sorted({op for op in op_types if op not in ESPDL_INT8_OPS})


def describe_op(op_type: str) -> str:
    """One line on an operator's status, including the known-artefact note."""
    if op_type in ESPDL_INT8_OPS:
        return f"{op_type}: supported (int8)"
    if op_type in KNOWN_EXPORT_ARTEFACTS:
        return f"{op_type}: NOT supported -- {KNOWN_EXPORT_ARTEFACTS[op_type]}"
    return f"{op_type}: NOT supported by ESP-DL int8"
