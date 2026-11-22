"""Named constants of the localization task (values come from the task PDF unless noted)."""

from __future__ import annotations

IMAGE_SIZE = 512
"""Side of every image, in pixels."""
GRID_SIZE = 128
"""Side of the evaluation grid; one cell is IMAGE_SIZE // GRID_SIZE = 4 pixels wide."""
CELL_WATERMARK_FRACTION = 0.5
"""A ground-truth cell is 1 when at least this fraction of its pixels is watermarked."""
MASK_WATERMARKED = 1
MASK_CLEAN = 0
CLASS_VALUES: tuple[int, int] = (MASK_CLEAN, MASK_WATERMARKED)

METHOD_PIXELSEAL = "pixelseal"
METHOD_TRUSTMARK = "trustmark"
METHOD_STEGASTAMP = "stegastamp"
KNOWN_METHODS: tuple[str, ...] = (METHOD_PIXELSEAL, METHOD_TRUSTMARK, METHOD_STEGASTAMP)
MAX_ASPECT_RATIO_BY_METHOD: dict[str, float] = {
    METHOD_PIXELSEAL: 16.0,
    METHOD_STEGASTAMP: 16.0,
    METHOD_TRUSTMARK: 2.0,
}
"""Largest rectangle aspect ratio the task authors use for each method."""

PIXELSEAL_N_BITS = 256
PIXELSEAL_NATIVE_SIZE = 256
"""Pixel Seal resizes every input to 256x256 before its extractor (see wmkit.seal)."""

SUBMISSION_KEY_NAMES = "image_names"
SUBMISSION_KEY_MASKS = "masks"
MAX_SUBMISSION_BYTES = 20_000_000
"""The PDF says 20 MB; the decimal value is the stricter reading, so a file that passes here
passes under either definition."""
SUBMISSION_DTYPE = "uint8"

DEFAULT_REPO_ID = "SprintML/Hackathon_Watermark_Localisation_Task"
FALLBACK_REPO_IDS: tuple[str, ...] = ("SprintML/Hackathon_Watermark_Localisation_Task_Paris",)
"""The PDF links the `_Paris` repo, the official JURECA script downloads the one without it."""
ARCHIVE_TARGETS: dict[str, str] = {
    "starter_kit.zip": "starter_kit",
    "train.zip": "release",
    "test.zip": "release",
}
"""Folder (under the destination) each archive unpacks into, as in the official README:
train.zip and test.zip hold `train/...` and `test/...` and go to `release/`, while
starter_kit.zip holds bare files (`wmloc/`, `download_models.py`, ...) and gets its own folder."""
KIT_ARCHIVES: tuple[str, ...] = tuple(ARCHIVE_TARGETS)
STARTER_KIT_SUBDIR = "starter_kit"
RELEASE_SUBDIR = "release"
KIT_TASK_ID = "34-watermark-localisation-image"
"""Server task id, from the official submission_template.py (the PDF does not give it)."""
WMLOC_DATA_SUBDIR = "wmloc"
EXTRACTED_MARKER_SUFFIX = ".extracted"
HTTP_NOT_FOUND = 404
HTTP_FORBIDDEN = 403
HTTP_UNAUTHORIZED = 401

DEFAULT_SEED = 2026
"""Seed of every synthetic split and of the bootstrap resampling."""
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_CONFIDENCE = 0.95
