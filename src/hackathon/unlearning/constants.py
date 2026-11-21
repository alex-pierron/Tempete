"""Constants of the unlearning task (values copied from the official task_template.py)."""

from __future__ import annotations

from pathlib import Path

TASK_ID = "31-unlearning"
HF_DATASET_REPO = "SprintML/unlearning"
"""Public Hugging Face dataset holding model.pt and forget_images.npz."""

MODEL_FILENAME = "model.pt"
FORGET_IMAGES_FILENAME = "forget_images.npz"
FORGET_IMAGES_KEY = "images"
SUBMISSION_FILENAME = "submission.pt"
DATA_SUBDIR = "unlearning"
"""Folder under the hackathon data directory (HACKATHON_DATA_DIR) for this task."""
OUTPUT_SUBDIR = "output"
"""Folder under the task dir where `run` writes candidates (same name as the JURECA output/)."""

MODEL_NAME = "vit_base_patch16_224"
NUM_CLASSES = 100
LAYER_SCALE_INIT = 1e-4
IMAGE_SIZE = 224
IMAGE_CHANNELS = 3
EXPECTED_FORGET_IMAGES = 3500

PIXEL_MAX = 255.0
NORMALISATION_CENTER = 0.5
NORMALISATION_SCALE = 0.5
"""Server preprocessing: (pixels / 255 - 0.5) / 0.5, i.e. values in [-1, 1]."""

MODEL_SHA256 = "485881a03ad064b5e6defc245ae3f45fd96714bf065ff96543a5eeca9f642571"
MODEL_SIZE_BYTES = 343_617_075
BYTES_PER_MEGABYTE = 1_000_000
MAX_SUBMISSION_BYTES = 450 * BYTES_PER_MEGABYTE
"""Official limit: "at most 450 MB". Decimal megabytes, the stricter reading."""
ALLOWED_SUFFIXES = (".pt", ".pth")

SCORING_TEMPERATURE = 2.0
"""Softmax temperature of every distribution in the server score (p_m = softmax(z / 2))."""
DEFAULT_BATCH_SIZE = 64
"""Forward batch size; ViT-B/16 in fp32 stays well under 16 GB at 64 on CPU/MPS and A100."""
HASH_CHUNK_BYTES = 1 << 20

PROXY_IMAGE_DIRS = ("coco_val_subset", "div2k_valid", "kodak")
"""Folders of `<repo>/data` used as an out-of-distribution retain proxy."""
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
DEFAULT_SEED = 0
"""Named seed of every random draw (subsets, alterations, training)."""


def default_task_dir(data_dir: Path) -> Path:
    """Directory holding the task files under the hackathon data directory."""
    return data_dir / DATA_SUBDIR
