"""Tests of wmkit.batch and of the `batch-embed` / `batch-extract` commands.

Fast tests use the classical methods (`dct_qim`, `spread_spectrum`) on tiny images, on CPU
unless stated. Tests marked `slow` load the real Pixel Seal checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from conftest import load_sample
from PIL import Image

from wmkit import batch as batch_module
from wmkit.batch import (
    FALLBACK_BATCH_SIZE,
    MAX_BATCH_SIZE,
    MIN_BATCH_SIZE,
    BatchError,
    BatchPlan,
    auto_batch_size,
    embed_folder,
    embed_images,
    extract_folder,
    extract_images,
    plan_batches,
    probe_image_shape,
    read_expected_messages,
    resolve_batch_size,
    score_extractions,
)
from wmkit.cli import main
from wmkit.io import load_image, save_image
from wmkit.messages import bits_to_hex, random_messages
from wmkit.methods import get_watermarker
from wmkit.types import Watermarker

FAST_METHODS = ["dct_qim", "spread_spectrum"]
SQUARE_SIDE = 64
WIDE_WIDTH = 96
CARRIER_SIDE = 128
CARRIER_WIDTH = 160
"""Both classical methods read their 32 bits back perfectly only from images of at least
~128 px (at 64x64 they reach 81 % bit accuracy even one image at a time), so the tests that
assert payload recovery use these sizes; the pure batch-vs-single tests also use 64x64."""
N_IMAGES = 7
MESSAGE_SEED = 5
EMBED_ABSOLUTE_TOLERANCE = 1e-5
"""Batch and one-by-one embedding run the same math on the same inputs; the only freedom is
the float32 summation order of the batched transforms (a few ulp, ~1e-7 on values in [0, 1]).
1e-5 leaves a 100x margin yet stays 400x below one 8-bit grey level (3.9e-3), so any real
divergence (wrong message routed to an image, shape mix-up) still fails."""
SOFT_BIT_ABSOLUTE_TOLERANCE = 1e-4
BATCH_SIZES = [1, 2, 3, N_IMAGES, 64]
SLOW_BATCH_SIZE = 4
SLOW_MIN_BIT_ACC = 0.95

SAMPLE_NAMES = ("astronaut", "coffee", "chelsea", "rocket")


def make_images(shapes: list[tuple[int, int]]) -> list[torch.Tensor]:
    """Distinct natural crops of the given `(H, W)` shapes (cycling over the samples)."""
    return [
        load_sample(SAMPLE_NAMES[index % len(SAMPLE_NAMES)], shape)
        for index, shape in enumerate(shapes)
    ]


def mixed_shapes() -> list[tuple[int, int]]:
    """64x64 and 64x96 images, interleaved so that groups are not contiguous."""
    return [
        (SQUARE_SIDE, SQUARE_SIDE) if index % 3 else (SQUARE_SIDE, WIDE_WIDTH)
        for index in range(N_IMAGES)
    ]


def hard(soft_bits: torch.Tensor) -> torch.Tensor:
    return (soft_bits > 0.5).to(torch.int64)


@pytest.fixture(scope="module", params=FAST_METHODS)
def watermarker(request: pytest.FixtureRequest) -> Watermarker:
    method: str = request.param
    return get_watermarker(method, device="cpu")


@pytest.fixture(autouse=True)
def cpu_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI tests run on CPU: deterministic and independent of the host GPU."""
    monkeypatch.setenv("WMKIT_DEVICE", "cpu")


@pytest.fixture()
def mixed_dir(tmp_path: Path) -> Path:
    """Folder of N_IMAGES lossless images, two interleaved sizes (128x128 and 128x160)."""
    folder = tmp_path / "mixed"
    shapes = [(CARRIER_SIDE, CARRIER_SIDE) if index % 3 else (CARRIER_SIDE, CARRIER_WIDTH)
              for index in range(N_IMAGES)]
    for index, image in enumerate(make_images(shapes)):
        save_image(image, folder / f"img_{index:02d}.png")
    return folder


# --- equivalence: batch vs one by one -------------------------------------------------


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_extract_batched_is_bit_exact_with_one_by_one(
    watermarker: Watermarker, batch_size: int
) -> None:
    images = make_images([(CARRIER_SIDE, CARRIER_SIDE)] * N_IMAGES)
    messages = random_messages(N_IMAGES, watermarker.n_bits, torch.Generator().manual_seed(1))
    marked = embed_images(watermarker, images, messages, batch_size=1)

    one_by_one = torch.cat([watermarker.extract(image[None]) for image in marked])
    batched = extract_images(watermarker, marked, batch_size=batch_size)

    assert torch.equal(hard(batched), hard(one_by_one))
    assert torch.allclose(batched, one_by_one, atol=SOFT_BIT_ABSOLUTE_TOLERANCE)
    # The batch really carries the payloads (not just equal garbage).
    assert torch.equal(hard(batched), messages.to(torch.int64))


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_embed_batched_matches_one_by_one(watermarker: Watermarker, batch_size: int) -> None:
    images = make_images([(SQUARE_SIDE, SQUARE_SIDE)] * N_IMAGES)
    messages = random_messages(N_IMAGES, watermarker.n_bits, torch.Generator().manual_seed(2))

    one_by_one = [watermarker.embed(image[None], messages[i:i + 1])[0]
                  for i, image in enumerate(images)]
    batched = embed_images(watermarker, images, messages, batch_size=batch_size)

    assert len(batched) == N_IMAGES
    for expected, actual in zip(one_by_one, batched, strict=True):
        assert torch.allclose(actual, expected, atol=EMBED_ABSOLUTE_TOLERANCE, rtol=0.0)


def test_equivalence_holds_with_heterogeneous_sizes(watermarker: Watermarker) -> None:
    images = make_images(mixed_shapes())
    messages = random_messages(N_IMAGES, watermarker.n_bits, torch.Generator().manual_seed(3))

    batched = embed_images(watermarker, images, messages, batch_size=2)
    for index, (original, result) in enumerate(zip(images, batched, strict=True)):
        assert result.shape == original.shape
        expected = watermarker.embed(original[None], messages[index:index + 1])[0]
        assert torch.allclose(result, expected, atol=EMBED_ABSOLUTE_TOLERANCE, rtol=0.0)

    extracted = extract_images(watermarker, batched, batch_size=2)
    one_by_one = torch.cat([watermarker.extract(image[None]) for image in batched])
    assert torch.equal(hard(extracted), hard(one_by_one))


def test_embed_images_rejects_misrouted_inputs(watermarker: Watermarker) -> None:
    images = make_images([(SQUARE_SIDE, SQUARE_SIDE)] * 2)
    wrong = torch.zeros(3, watermarker.n_bits)
    with pytest.raises(BatchError, match="messages must have shape"):
        embed_images(watermarker, images, wrong)
    with pytest.raises(BatchError, match="no image"):
        extract_images(watermarker, [])


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs the MPS backend")
def test_extract_batched_is_bit_exact_on_mps() -> None:
    mps_watermarker = get_watermarker("dct_qim", device="mps")
    images = make_images([(CARRIER_SIDE, CARRIER_SIDE)] * N_IMAGES)
    messages = random_messages(N_IMAGES, mps_watermarker.n_bits, torch.Generator().manual_seed(4))
    marked = embed_images(mps_watermarker, images, messages, batch_size=3, device="mps")
    batched = extract_images(mps_watermarker, marked, batch_size=3, device="mps")
    single = extract_images(mps_watermarker, marked, batch_size=1, device="mps")
    assert torch.equal(hard(batched), hard(single))
    assert torch.equal(hard(batched), messages.to(torch.int64))


# --- planning: grouping by shape, batch size -------------------------------------------


def test_plan_groups_by_shape_and_keeps_order() -> None:
    wide = (SQUARE_SIDE, WIDE_WIDTH)
    square = (SQUARE_SIDE, SQUARE_SIDE)
    shapes = [wide, square, square, wide, square]
    plans = plan_batches(shapes, batch_size=2, device="cpu")
    assert plans == [
        BatchPlan(shape=wide, indices=(0, 3)),
        BatchPlan(shape=square, indices=(1, 2)),
        BatchPlan(shape=square, indices=(4,)),
    ]
    for plan in plans:
        assert {shapes[index] for index in plan.indices} == {plan.shape}


def test_plan_without_images_is_an_error() -> None:
    with pytest.raises(BatchError, match="no image"):
        plan_batches([], batch_size=None, device="cpu")


def test_explicit_batch_size_is_honoured_and_zero_means_auto() -> None:
    shape = (SQUARE_SIDE, SQUARE_SIDE)
    assert resolve_batch_size(5, "cpu", shape) == 5
    assert resolve_batch_size(0, "cpu", shape) == auto_batch_size("cpu", shape)
    assert resolve_batch_size(None, "cpu", shape) == auto_batch_size("cpu", shape)
    with pytest.raises(BatchError, match="batch size"):
        resolve_batch_size(-1, "cpu", shape)


def test_auto_batch_size_is_clamped_and_shrinks_with_image_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (SQUARE_SIDE, SQUARE_SIDE)
    gib = 1024**3
    monkeypatch.setattr(batch_module, "_available_memory_bytes", lambda device: 0)
    assert auto_batch_size("cpu", shape) == MIN_BATCH_SIZE
    monkeypatch.setattr(batch_module, "_available_memory_bytes", lambda device: 4096 * gib)
    assert auto_batch_size("cpu", shape) == MAX_BATCH_SIZE
    monkeypatch.setattr(batch_module, "_available_memory_bytes", lambda device: 8 * gib)
    small = auto_batch_size("mps", (256, 256))
    large = auto_batch_size("mps", (2048, 2048))
    assert MIN_BATCH_SIZE <= large < small <= MAX_BATCH_SIZE


def test_auto_batch_size_falls_back_without_a_memory_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(batch_module, "_available_memory_bytes", lambda device: None)
    assert auto_batch_size("mps", (512, 512)) == FALLBACK_BATCH_SIZE


def test_auto_batch_size_on_this_machine_is_within_bounds() -> None:
    assert MIN_BATCH_SIZE <= auto_batch_size("cpu", (512, 512)) <= MAX_BATCH_SIZE


def test_auto_batch_size_rejects_an_empty_shape() -> None:
    with pytest.raises(BatchError, match="positive"):
        auto_batch_size("cpu", (0, 64))


# --- the watermarker's own forward-pass cap ---------------------------------------------


class CappedWatermarker:
    """Stub with the `batch_size` cap of `SealWatermarker`; records the cap seen per call."""

    name = "capped"
    n_bits = 4

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.caps_seen: list[int] = []

    def embed(self, images: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        self.caps_seen.append(self.batch_size)
        return images.clone()

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        self.caps_seen.append(self.batch_size)
        return torch.full((images.shape[0], self.n_bits), 0.5)


def test_internal_forward_pass_cap_is_raised_for_the_run_then_restored() -> None:
    stub = CappedWatermarker(batch_size=2)
    images = make_images([(SQUARE_SIDE, SQUARE_SIDE)] * 6)
    extract_images(stub, images, batch_size=6)
    assert stub.caps_seen == [6]  # one call of 6 images, allowed to run as one pass
    assert stub.batch_size == 2


def test_internal_forward_pass_cap_is_restored_after_an_error() -> None:
    images = make_images([(SQUARE_SIDE, SQUARE_SIDE)] * 3)
    messages = torch.zeros(3, CappedWatermarker.n_bits)
    # Error raised INSIDE the context (extract returns a wrong shape): the cap must be restored.
    wrong_extract = _WrongShapeWatermarker(batch_size=2)
    with pytest.raises(BatchError, match="returned"):
        extract_images(wrong_extract, images, batch_size=3)
    assert wrong_extract.caps_seen == [3]  # the cap was actually raised to 3 before the failure
    assert wrong_extract.batch_size == 2  # restored by the finally, not only on the happy path
    # Same on the embed path: embed() itself raises inside the context.
    wrong_embed = _WrongShapeEmbedder(batch_size=2)
    with pytest.raises(BatchError, match="returned"):
        embed_images(wrong_embed, images, messages, batch_size=3)
    assert wrong_embed.caps_seen == [3]
    assert wrong_embed.batch_size == 2


class _WrongShapeWatermarker(CappedWatermarker):
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        self.caps_seen.append(self.batch_size)
        return torch.zeros(1, self.n_bits)


class _WrongShapeEmbedder(CappedWatermarker):
    def embed(self, images: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        self.caps_seen.append(self.batch_size)
        return images[:, :, :1, :1]  # wrong spatial shape on purpose


def test_a_larger_caller_cap_is_kept() -> None:
    stub = CappedWatermarker(batch_size=32)
    images = make_images([(SQUARE_SIDE, SQUARE_SIDE)] * 4)
    extract_images(stub, images, batch_size=4)
    assert stub.caps_seen == [32]
    assert stub.batch_size == 32


# --- folders ---------------------------------------------------------------------------


def test_embed_folder_then_extract_folder_round_trip(
    mixed_dir: Path, tmp_path: Path
) -> None:
    method = get_watermarker("dct_qim", device="cpu")
    out_dir = tmp_path / "marked"
    embedded = embed_folder(method, mixed_dir, out_dir, batch_size=3, seed=MESSAGE_SEED)

    assert len(embedded.records) == N_IMAGES
    assert set(embedded.batch_sizes) == {(CARRIER_SIDE, CARRIER_SIDE),
                                         (CARRIER_SIDE, CARRIER_WIDTH)}
    for record in embedded.records:
        assert record.output_path.is_file()
        original = load_image(mixed_dir / record.image_name)
        assert load_image(record.output_path).shape == original.shape

    extracted = extract_folder(method, out_dir, batch_size=2)
    assert extracted.image_names == tuple(f"img_{i:02d}.png" for i in range(N_IMAGES))
    assert [bits_to_hex(row) for row in extracted.soft_bits] == [
        record.message_hex for record in embedded.records
    ]


def test_embed_folder_output_does_not_depend_on_batch_size(
    mixed_dir: Path, tmp_path: Path
) -> None:
    method = get_watermarker("spread_spectrum", device="cpu")
    first = embed_folder(method, mixed_dir, tmp_path / "bs1", batch_size=1, seed=MESSAGE_SEED)
    second = embed_folder(method, mixed_dir, tmp_path / "bs4", batch_size=4, seed=MESSAGE_SEED)
    assert [r.message_hex for r in first.records] == [r.message_hex for r in second.records]
    for left, right in zip(first.records, second.records, strict=True):
        # Compared after 8-bit quantisation (what lands on disk): an ulp-level difference
        # may flip a pixel by one grey level at a rounding boundary, never more.
        difference = (load_image(left.output_path) - load_image(right.output_path)).abs().max()
        assert float(difference) <= 1.0 / 255.0 + 1e-6


def test_embed_folder_seed_changes_the_messages(mixed_dir: Path, tmp_path: Path) -> None:
    method = get_watermarker("dct_qim", device="cpu")
    first = embed_folder(method, mixed_dir, tmp_path / "s1", seed=1)
    again = embed_folder(method, mixed_dir, tmp_path / "s1b", seed=1)
    other = embed_folder(method, mixed_dir, tmp_path / "s2", seed=2)
    assert [r.message_hex for r in first.records] == [r.message_hex for r in again.records]
    assert [r.message_hex for r in first.records] != [r.message_hex for r in other.records]


def test_embed_folder_writes_png_even_for_jpeg_inputs(tmp_path: Path) -> None:
    folder = tmp_path / "jpegs"
    save_image(load_sample("astronaut", (SQUARE_SIDE, SQUARE_SIDE)), folder / "photo.jpg")
    method = get_watermarker("dct_qim", device="cpu")
    result = embed_folder(method, folder, tmp_path / "out")
    assert result.records[0].output_path == tmp_path / "out" / "photo.png"
    assert result.records[0].output_path.is_file()


def test_embed_folder_rejects_stem_collisions(tmp_path: Path) -> None:
    folder = tmp_path / "clash"
    image = load_sample("astronaut", (SQUARE_SIDE, SQUARE_SIDE))
    save_image(image, folder / "same.png")
    save_image(image, folder / "same.jpg")
    method = get_watermarker("dct_qim", device="cpu")
    with pytest.raises(BatchError, match="same output file"):
        embed_folder(method, folder, tmp_path / "out")


def test_empty_folder_is_a_clean_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    method = get_watermarker("dct_qim", device="cpu")
    with pytest.raises(BatchError, match="no image"):
        embed_folder(method, empty, tmp_path / "out")
    with pytest.raises(BatchError, match="no image"):
        extract_folder(method, empty)
    assert not (tmp_path / "out").exists()


def test_missing_folder_is_reported(tmp_path: Path) -> None:
    method = get_watermarker("dct_qim", device="cpu")
    with pytest.raises(NotADirectoryError):
        extract_folder(method, tmp_path / "nowhere")


def test_probe_image_shape_follows_exif_orientation(tmp_path: Path) -> None:
    plain = tmp_path / "plain.png"
    save_image(load_sample("astronaut", (SQUARE_SIDE, WIDE_WIDTH)), plain)
    assert probe_image_shape(plain) == (SQUARE_SIDE, WIDE_WIDTH)

    rotated = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (WIDE_WIDTH, SQUARE_SIDE), color=(120, 40, 200))
    exif = Image.Exif()
    exif[batch_module.EXIF_ORIENTATION_TAG] = 6  # rotate 90: width and height swap
    image.save(rotated, exif=exif)
    assert probe_image_shape(rotated) == tuple(load_image(rotated).shape[-2:])
    assert probe_image_shape(rotated) == (WIDE_WIDTH, SQUARE_SIDE)


def test_read_expected_messages_positional_and_named(tmp_path: Path) -> None:
    names = ["a.png", "b.png"]
    good = tmp_path / "good.txt"
    good.write_text("deadbeef\n\n00000001\n", encoding="utf-8")  # positional: one hex per line
    bits = read_expected_messages(good, n_bits=32, image_names=names)
    assert bits.shape == (2, 32)
    assert bits_to_hex(bits[0]) == "deadbeef"

    with pytest.raises(BatchError, match="2 expected message"):
        read_expected_messages(good, n_bits=32, image_names=["a.png", "b.png", "c.png"])
    bad = tmp_path / "bad.txt"
    bad.write_text("deadbeef\nxyz\n", encoding="utf-8")
    with pytest.raises(BatchError, match="line 2"):  # blames the real source line
        read_expected_messages(bad, n_bits=32, image_names=names)

    # 'name,hex' format: pairing follows the file name, not the order of the lines.
    named = tmp_path / "named.txt"
    named.write_text("b.png,00000001\na.png,deadbeef\n", encoding="utf-8")  # reversed on purpose
    reordered = read_expected_messages(named, n_bits=32, image_names=names)
    assert bits_to_hex(reordered[0]) == "deadbeef"  # row 0 is a.png, though written second
    assert bits_to_hex(reordered[1]) == "00000001"
    with pytest.raises(BatchError, match="missing"):
        read_expected_messages(named, n_bits=32, image_names=["a.png", "c.png"])


def test_embed_folder_refuses_to_overwrite_inputs_or_collide(tmp_path: Path) -> None:
    watermarker = get_watermarker("dct_qim", device="cpu")
    images_dir = tmp_path / "in"
    images_dir.mkdir()
    save_image(torch.rand(3, SQUARE_SIDE, SQUARE_SIDE), images_dir / "one.png")
    # out_dir == images_dir would write the marked images back over the inputs.
    with pytest.raises(BatchError, match="must differ from images_dir"):
        embed_folder(watermarker, images_dir, images_dir)
    # Outputs that collide only by case must be caught (APFS/NTFS are case-insensitive):
    # 'A.jpg' -> 'A.png' and 'a.png' -> 'a.png' are one file on those systems.
    save_image(torch.rand(3, SQUARE_SIDE, SQUARE_SIDE), images_dir / "A.jpg")
    save_image(torch.rand(3, SQUARE_SIDE, SQUARE_SIDE), images_dir / "a.png")
    with pytest.raises(BatchError, match="same output file"):
        embed_folder(watermarker, images_dir, tmp_path / "out")


def test_score_extractions_flags_detection() -> None:
    n_bits = 32
    truth = random_messages(2, n_bits, torch.Generator().manual_seed(9))
    soft = truth.clone()
    soft[1] = 1.0 - soft[1]  # second image reads the complement: every bit wrong
    perfect, inverted = score_extractions(soft, truth)
    assert perfect.bit_accuracy == 1.0 and perfect.detected
    assert inverted.bit_accuracy == 0.0 and not inverted.detected


# --- CLI -------------------------------------------------------------------------------


def run_cli(capsys: pytest.CaptureFixture[str], *arguments: str) -> tuple[int, str, str]:
    status = main(list(arguments))
    captured = capsys.readouterr()
    return status, captured.out, captured.err


def csv_rows(output: str, header: str) -> list[list[str]]:
    lines = output.strip().splitlines()
    start = lines.index(header)
    return [line.split(",") for line in lines[start + 1:] if not line.startswith("summary:")]


@pytest.mark.parametrize("batch_arguments", [[], ["--batch-size", "0"], ["--batch-size", "3"]])
def test_cli_batch_embed_then_extract_with_expected(
    capsys: pytest.CaptureFixture[str], mixed_dir: Path, tmp_path: Path,
    batch_arguments: list[str],
) -> None:
    out_dir = tmp_path / "marked"
    messages_file = tmp_path / "messages.txt"
    status, out, err = run_cli(
        capsys, "batch-embed", "--method", "dct_qim", "--images", str(mixed_dir),
        "--out", str(out_dir), "--seed", "7", "--messages-out", str(messages_file),
        *batch_arguments,
    )
    assert status == 0, err
    assert f"{N_IMAGES} image(s) watermarked" in out
    rows = csv_rows(out, "image,message_hex")
    assert [row[0] for row in rows] == [f"img_{i:02d}.png" for i in range(N_IMAGES)]
    assert all(len(row[1]) == 8 for row in rows)  # 32 bits
    # --messages-out now writes 'output_name,hex' (inputs are .png, so output name == input name).
    assert messages_file.read_text(encoding="utf-8").split() == [f"{row[0]},{row[1]}" for row in rows]
    assert len(list(out_dir.iterdir())) == N_IMAGES

    status, out, err = run_cli(
        capsys, "batch-extract", "--method", "dct_qim", "--images", str(out_dir),
        "--expected", str(messages_file), *batch_arguments,
    )
    assert status == 0, err
    scored = csv_rows(out, "image,bits,bit_acc,neg_log10_p,detected")
    assert [row[1] for row in scored] == [row[1] for row in rows]
    assert all(row[2] == "1.0000" and row[4] == "True" for row in scored)
    assert f"summary: images={N_IMAGES} mean_bit_acc=1.0000" in out
    assert f"detected={N_IMAGES}/{N_IMAGES}" in out


def test_cli_batch_extract_without_expected_prints_bits(
    capsys: pytest.CaptureFixture[str], mixed_dir: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "marked"
    run_cli(capsys, "batch-embed", "--method", "spread_spectrum", "--images", str(mixed_dir),
            "--out", str(out_dir))
    status, out, _ = run_cli(capsys, "batch-extract", "--method", "spread_spectrum",
                             "--images", str(out_dir), "--batch-size", "2")
    assert status == 0
    rows = csv_rows(out, "image,bits")
    assert len(rows) == N_IMAGES
    assert "bit_acc" not in out
    assert "batch size 2" in out


def test_cli_batch_embed_default_matches_single_embed_message(
    capsys: pytest.CaptureFixture[str], mixed_dir: Path, tmp_path: Path
) -> None:
    """A folder of one image gets the same message as `embed --seed` (same generator)."""
    folder = tmp_path / "one"
    save_image(load_sample("coffee", (SQUARE_SIDE, SQUARE_SIDE)), folder / "only.png")
    _, batch_out, _ = run_cli(capsys, "batch-embed", "--method", "dct_qim", "--images",
                              str(folder), "--out", str(tmp_path / "b"), "--seed", "11")
    _, single_out, _ = run_cli(capsys, "embed", "--method", "dct_qim", "--in",
                               str(folder / "only.png"), "--out", str(tmp_path / "s.png"),
                               "--seed", "11")
    assert csv_rows(batch_out, "image,message_hex")[0][1] == single_out.strip()


@pytest.mark.parametrize("command", ["batch-embed", "batch-extract"])
def test_cli_empty_folder_fails_cleanly(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, command: str
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    arguments = [command, "--method", "dct_qim", "--images", str(empty)]
    if command == "batch-embed":
        arguments += ["--out", str(tmp_path / "out")]
    status, out, err = run_cli(capsys, *arguments)
    assert status == 1
    assert out == ""
    assert "error: no image" in err
    assert "Traceback" not in err


def test_cli_rejects_negative_batch_size_and_bad_expected_file(
    capsys: pytest.CaptureFixture[str], mixed_dir: Path, tmp_path: Path
) -> None:
    status, _, err = run_cli(capsys, "batch-extract", "--method", "dct_qim",
                             "--images", str(mixed_dir), "--batch-size", "-2")
    assert status == 1
    assert "batch size must be >= 0" in err

    short = tmp_path / "short.txt"
    short.write_text("deadbeef\n", encoding="utf-8")
    status, _, err = run_cli(capsys, "batch-extract", "--method", "dct_qim",
                             "--images", str(mixed_dir), "--expected", str(short))
    assert status == 1
    assert "expected message" in err


def test_cli_unknown_method_and_usage_errors(
    capsys: pytest.CaptureFixture[str], mixed_dir: Path, tmp_path: Path
) -> None:
    status, _, err = run_cli(capsys, "batch-extract", "--method", "nope",
                             "--images", str(mixed_dir))
    assert status == 1
    assert "unknown watermarking method" in err
    with pytest.raises(SystemExit) as exit_info:
        main(["batch-embed", "--method", "dct_qim", "--images", str(mixed_dir)])  # no --out
    assert exit_info.value.code == 2


# --- slow: the real model --------------------------------------------------------------


@pytest.mark.slow
def test_pixelseal_batched_matches_one_by_one() -> None:
    """Real Pixel Seal: batched bits equal the one-by-one bits and carry the payload."""
    model = get_watermarker("pixelseal", device="cpu")
    images = make_images([(SQUARE_SIDE * 2, SQUARE_SIDE * 2)] * SLOW_BATCH_SIZE)
    messages = random_messages(SLOW_BATCH_SIZE, model.n_bits, torch.Generator().manual_seed(6))
    marked = embed_images(model, images, messages, batch_size=SLOW_BATCH_SIZE)
    batched = extract_images(model, marked, batch_size=SLOW_BATCH_SIZE)
    single = extract_images(model, marked, batch_size=1)
    assert torch.equal(hard(batched), hard(single))
    accuracy = (hard(batched) == messages.to(torch.int64)).to(torch.float32).mean()
    assert float(accuracy) >= SLOW_MIN_BIT_ACC
