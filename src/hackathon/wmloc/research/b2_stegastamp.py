"""Spatial features of the StegaStamp decoder, tapped from its ONNX graph (team B2).

The official wrapper only exposes the 100 decoded logits: the decoder ends with dense layers over a
flattened map, so there is no per-position head like Pixel Seal's or TrustMark's. The convolutional
layers before it are still local, and the head that localised Pixel Seal from its ConvNeXt stages
can read them the same way. We add three intermediate activations as graph outputs of a copy of
the kit's ONNX file (the released graph only outputs rounded bits; the kit already does the same
for the pre-sigmoid logits) and pool the two high-resolution ones to a 25 x 25 grid.

Cached arrays per image, in the same file layout as `b2_features`:
`ss_c3` [64, 25, 25] (layer 3, 100 x 100 pooled by 4), `ss_c5` [64, 25, 25] (layer 5, 50 x 50 pooled
by 2), `ss_c7` [128, 13, 13] (layer 7), `scalars` [3] (global margin, agreement, spread).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.nn import functional

from hackathon.wmloc.research.b2_features import (
    MARGIN_CLIP,
    cache_root,
    message_signs,
)
from hackathon.wmloc.research.errors import ResearchError

STEGASTAMP_SIDE = 400
"""Working resolution of the StegaStamp decoder (bicubic resize, as the kit does)."""
LOGITS_NAME = "gen_decoder/decoder_out/BiasAdd:0"
FEATURE_TENSORS: dict[str, tuple[str, int]] = {
    "ss_c3": ("gen_decoder/dec_conv_3/Relu:0", 4),
    "ss_c5": ("gen_decoder/dec_conv_5/Relu:0", 2),
    "ss_c7": ("gen_decoder/dec_conv_7/Relu:0", 1),
}
"""Source name -> (graph tensor, average-pooling factor applied to the activation)."""
FEATURE_GRIDS: dict[str, int] = {"ss_c3": 25, "ss_c5": 25, "ss_c7": 13}
"""Side of each cached map (layer 3: 100 / 4, layer 5: 50 / 2, layer 7: 13 as produced)."""
ONNX_THREADS = 4
MODEL_FILE = "stegastamp_feats.onnx"


class StegaStampFeatureError(ResearchError):
    """The kit's ONNX file is missing or the tapped graph does not behave as expected."""


def feature_model_path() -> Path:
    """Where the augmented ONNX graph is written (next to the feature cache)."""
    return cache_root().parent / MODEL_FILE


def ensure_feature_model(kit_onnx: Path) -> Path:
    """Write (once) a copy of `kit_onnx` that also outputs the tapped convolution activations.

    Raises:
        StegaStampFeatureError: the kit file does not exist.
    """
    target = feature_model_path()
    if target.is_file():
        return target
    if not kit_onnx.is_file():
        raise StegaStampFeatureError(f"missing {kit_onnx}")
    import onnx  # kit venv only

    model = onnx.load(str(kit_onnx))
    present = {output.name for output in model.graph.output}
    for tensor, _ in FEATURE_TENSORS.values():
        if tensor not in present:
            info = onnx.helper.make_tensor_value_info(tensor, onnx.TensorProto.FLOAT, None)
            model.graph.output.append(info)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(target))
    return target


class StegaStampProbe:
    """Runs the augmented decoder graph and returns the pooled feature maps per image."""

    def __init__(self, kit_onnx: Path) -> None:
        import onnxruntime as ort  # kit venv only

        options = ort.SessionOptions()
        options.intra_op_num_threads = ONNX_THREADS
        self.session = ort.InferenceSession(
            str(ensure_feature_model(kit_onnx)), options, providers=["CPUExecutionProvider"]
        )
        self.tensors = [LOGITS_NAME, *(tensor for tensor, _ in FEATURE_TENSORS.values())]

    def extract(self, images: torch.Tensor, messages: NDArray[np.float32]) -> list[dict[str, Any]]:
        """Feature dictionaries (float16) of a batch of [B, 3, 512, 512] images in [0, 1]."""
        resized = functional.interpolate(
            images.float().cpu(),
            size=(STEGASTAMP_SIDE,) * 2,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).clamp(0, 1)
        batch = resized.permute(0, 2, 3, 1).numpy().astype(np.float32)
        outputs = self.session.run(
            self.tensors, {"image": batch, "secret": np.zeros((len(batch), 100), np.float32)}
        )
        logits = torch.from_numpy(np.asarray(outputs[0])).float()
        rows: list[dict[str, Any]] = []
        for index in range(len(batch)):
            row: dict[str, Any] = {}
            for (name, (_, factor)), raw in zip(FEATURE_TENSORS.items(), outputs[1:], strict=True):
                maps = torch.from_numpy(np.asarray(raw[index])).float()
                if factor > 1:
                    maps = functional.avg_pool2d(maps[None], factor)[0]
                if maps.shape[-1] != FEATURE_GRIDS[name]:
                    raise StegaStampFeatureError(
                        f"{name}: expected a {FEATURE_GRIDS[name]} grid, got {tuple(maps.shape)}"
                    )
                row[name] = maps.numpy().astype(np.float16)
            signed = logits[index] * message_signs(messages[index])
            row["scalars"] = np.array(
                [
                    float(signed.mean().clamp(-MARGIN_CLIP, MARGIN_CLIP)),
                    float((signed > 0).float().mean()),
                    float(signed.std()),
                ],
                dtype=np.float32,
            )
            rows.append(row)
        return rows
