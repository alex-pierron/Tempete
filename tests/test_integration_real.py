"""Integration (slow): the merged package with the real pretrained weights and real data.

Everything here is marked `slow` (run with `-m slow`) and skips itself when the Kodak images, a
checkpoint or the cached language model are not on disk. Nothing is downloaded: the script tests
replace the network with a function that fails loudly.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch

from wmkit.attacks import attack_seed, get_attack, get_suite, reseed_attack
from wmkit.config import data_dir, weights_dir
from wmkit.eval import evaluate
from wmkit.io import list_images, load_image
from wmkit.messages import random_messages
from wmkit.methods import get_watermarker
from wmkit.metrics import DETECTION_NEG_LOG10_P, bit_accuracy, neg_log10_pvalue, psnr, ssim

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parents[1]
KODAK_COUNT = 24
PIXELSEAL_BITS = 256
VIDEOSEAL_BITS = 256
VIDEOSEAL0_BITS = 96
CLEAN_MIN_BIT_ACC = 0.99
CLEAN_MIN_PSNR_DB = 40.0
ROBUST_MIN_BIT_ACC = 0.90
ROBUST_ATTACKS = [
    "brightness_1.5", "contrast_0.5", "hue_0.1", "gaussian_blur_9", "jpeg_50", "crop_0.71",
    "resize_0.5", "rotate_10", "hflip",
]
SUBPROCESS_TIMEOUT_S = 600
TEXT_TOKENS = 90


def _kodak() -> list[Path]:
    paths = list_images(data_dir() / "kodak")
    if len(paths) < KODAK_COUNT:
        pytest.skip("Kodak images are not on disk (run scripts/fetch_data.py --dataset kodak)")
    return paths


def _checkpoint(card_file: str) -> Path:
    path = weights_dir() / "seal" / card_file
    if not path.exists():
        pytest.skip(f"{card_file} is not cached in {weights_dir()}")
    return path


@pytest.fixture(scope="module")
def pixelseal() -> Any:
    _checkpoint("pixelseal.slim.pth")
    return get_watermarker("pixelseal")


def _fingerprint(directory: Path) -> dict[str, tuple[int, int]]:
    return {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(directory.iterdir())}


class TestRealPixelSeal:
    def test_registry_resolves_the_real_cards(self) -> None:
        for method, n_bits, file_name in (
            ("pixelseal", PIXELSEAL_BITS, "pixelseal.slim.pth"),
            ("videoseal", VIDEOSEAL_BITS, "videoseal_1.0.slim.pth"),
            ("videoseal0", VIDEOSEAL0_BITS, "videoseal_0.0.slim.pth"),
        ):
            _checkpoint(file_name)
            watermarker = get_watermarker(method)
            assert watermarker.n_bits == n_bits, method
            assert watermarker.name

    def test_n_bits_must_be_native(self) -> None:
        _checkpoint("pixelseal.slim.pth")
        with pytest.raises(Exception, match="256|n_bits|native"):
            get_watermarker("pixelseal", n_bits=64)

    def test_kodak_round_trip_quality_and_detection(self, pixelseal: Any) -> None:
        image = load_image(_kodak()[0])[None]
        message = random_messages(1, pixelseal.n_bits, torch.Generator().manual_seed(0))
        with torch.no_grad():
            marked = pixelseal.embed(image, message)
            soft = pixelseal.extract(marked)
        assert marked.shape == image.shape and soft.shape == (1, PIXELSEAL_BITS)
        assert float(marked.min()) >= 0.0 and float(marked.max()) <= 1.0
        accuracy = bit_accuracy(soft, message)
        assert float(accuracy[0]) >= CLEAN_MIN_BIT_ACC
        score = neg_log10_pvalue(PIXELSEAL_BITS, torch.round(accuracy * PIXELSEAL_BITS))
        assert float(score[0]) > DETECTION_NEG_LOG10_P
        assert float(psnr(marked, image)[0]) >= CLEAN_MIN_PSNR_DB
        assert float(ssim(marked, image)[0]) >= 0.97

    def test_unmarked_image_is_not_detected(self, pixelseal: Any) -> None:
        image = load_image(_kodak()[1])[None]
        message = random_messages(1, pixelseal.n_bits, torch.Generator().manual_seed(1))
        with torch.no_grad():
            accuracy = bit_accuracy(pixelseal.extract(image), message)
        score = neg_log10_pvalue(PIXELSEAL_BITS, torch.round(accuracy * PIXELSEAL_BITS))
        assert float(score[0]) < DETECTION_NEG_LOG10_P

    @pytest.mark.parametrize("attack_name", ROBUST_ATTACKS)
    def test_registry_attack_then_extract_stays_detected(
        self, pixelseal: Any, attack_name: str
    ) -> None:
        image = load_image(_kodak()[2])[None]
        message = random_messages(1, pixelseal.n_bits, torch.Generator().manual_seed(2))
        spec = get_attack(attack_name)
        with torch.no_grad():
            marked = pixelseal.embed(image, message)
            reseed_attack(spec, attack_seed(0, "kodim03", attack_name))
            attacked = spec.fn(marked)
            accuracy = bit_accuracy(pixelseal.extract(attacked), message)
        assert float(accuracy[0]) >= ROBUST_MIN_BIT_ACC, (attack_name, float(accuracy[0]))

    def test_extract_logits_is_differentiable_and_consistent_with_extract(
        self, pixelseal: Any
    ) -> None:
        image = load_image(_kodak()[3])[None, :, :256, :256].clone().requires_grad_(True)
        logits = pixelseal.extract_logits(image)
        assert logits.shape == (1, PIXELSEAL_BITS)
        logits.sum().backward()
        assert image.grad is not None and float(image.grad.abs().sum()) > 0.0
        with torch.no_grad():
            soft = pixelseal.extract(image.detach())
        assert torch.allclose(torch.sigmoid(logits.detach()), soft, atol=1e-5)

    def test_harness_on_two_kodak_images(self, pixelseal: Any) -> None:
        images = [(p.name, load_image(p)) for p in _kodak()[:2]]
        suite = [s for s in get_suite("quick") if s.name in ("identity", "jpeg_50", "crop_0.71")]
        frame = evaluate(pixelseal, images, suite, seed=0)
        assert len(frame) == 2 * 3 * 2
        identity = frame[(frame["attack"] == "identity")]
        assert identity[identity["is_watermarked"]]["detected"].all()
        assert not frame[~frame["is_watermarked"]]["detected"].any()
        assert (frame["psnr"] >= CLEAN_MIN_PSNR_DB).all()
        assert frame["bit_acc"].between(0.0, 1.0).all()
        assert frame["lpips"].isna().all()
        assert not frame.drop(columns="lpips").isna().any().any()

    def test_cli_subprocess_embed_extract_on_kodak(self, tmp_path: Path) -> None:
        _checkpoint("pixelseal.slim.pth")
        source = _kodak()[4]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONWARNINGS"] = "ignore"

        def run(*args: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                [sys.executable, "-m", "wmkit", *args], capture_output=True, text=True, env=env,
                timeout=SUBPROCESS_TIMEOUT_S, cwd=tmp_path, check=False,
            )

        embedded = run("embed", "--method", "pixelseal", "--in", str(source),
                       "--out", str(tmp_path / "m.png"), "--seed", "3")
        assert embedded.returncode == 0, embedded.stderr
        message = embedded.stdout.strip().splitlines()[-1]
        assert len(message) == PIXELSEAL_BITS // 4
        extracted = run("extract", "--method", "pixelseal", "--in", str(tmp_path / "m.png"),
                        "--message", message)
        assert extracted.returncode == 0, extracted.stderr
        bit_acc = float(next(x for x in extracted.stdout.splitlines()
                             if x.startswith("bit_acc")).split(":")[1])
        assert bit_acc >= CLEAN_MIN_BIT_ACC
        assert "detected: True" in extracted.stdout

    def test_video_helpers_round_trip(self, pixelseal: Any) -> None:
        frames = torch.stack([load_image(p)[:, :256, :256] for p in _kodak()[:4]])
        message = random_messages(1, pixelseal.n_bits, torch.Generator().manual_seed(4))[0]
        with torch.no_grad():
            marked = pixelseal.embed_video(frames, message)
            decoded = pixelseal.extract_video(marked)
        assert marked.shape == frames.shape
        assert decoded.shape[-1] == PIXELSEAL_BITS
        accuracy = bit_accuracy(decoded.reshape(1, -1), message[None])
        assert float(accuracy[0]) >= ROBUST_MIN_BIT_ACC


class TestRealLanguageModel:
    def test_kgw_generation_and_detection_with_smollm(self) -> None:
        from wmkit.text import (
            DEFAULT_LM,
            KGWConfig,
            generate,
            kgw_detect,
            load_lm,
            model_vocab_size,
        )

        try:
            model, tokenizer = load_lm(DEFAULT_LM, device="cpu")
        except OSError:
            pytest.skip(f"{DEFAULT_LM} is not in the local Hugging Face cache")
        vocab = model_vocab_size(model)
        config = KGWConfig()
        prompt = "Write a short story about a lighthouse keeper."
        marked = generate(model, tokenizer, prompt, scheme="kgw", config=config,
                          max_new_tokens=TEXT_TOKENS, seed=0, min_new_tokens=TEXT_TOKENS)
        plain = generate(model, tokenizer, prompt, scheme="none", config=None,
                         max_new_tokens=TEXT_TOKENS, seed=0, min_new_tokens=TEXT_TOKENS)
        assert len(marked) == TEXT_TOKENS and all(0 <= t < vocab for t in marked)
        assert kgw_detect(marked, vocab, config).detected
        assert not kgw_detect(plain, vocab, config).detected


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_script_{name}", REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _network_forbidden(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    attempts: list[str] = []

    def refuse(request: Any, *args: Any, **kwargs: Any) -> Any:
        attempts.append(getattr(request, "full_url", str(request)))
        raise OSError("network is forbidden in this test")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    return attempts


class TestScriptsAreIdempotent:
    def test_fetch_data_kodak_does_not_touch_files_nor_network(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _kodak()
        folder = data_dir() / "kodak"
        before = _fingerprint(folder)
        attempts = _network_forbidden(monkeypatch)
        script = _load_script("fetch_data")
        assert script.main(["--dataset", "kodak"]) == 0
        assert script.main(["--dataset", "kodak"]) == 0
        assert attempts == []
        assert _fingerprint(folder) == before
        assert f"kodak: {KODAK_COUNT} images" in capsys.readouterr().out
        assert not list(folder.glob("*.part"))

    def test_fetch_data_rejects_an_unknown_dataset(self) -> None:
        script = _load_script("fetch_data")
        with pytest.raises(SystemExit) as excinfo:
            script.main(["--dataset", "imagenet"])
        assert excinfo.value.code == 2

    def test_fetch_weights_with_cached_cards_does_not_download(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cards = {"pixelseal": "pixelseal.slim.pth", "videoseal": "videoseal_1.0.slim.pth",
                 "videoseal0": "videoseal_0.0.slim.pth"}
        folder = weights_dir() / "seal"
        for file_name in cards.values():
            _checkpoint(file_name)
        before = {name: (folder / name).stat().st_mtime_ns for name in cards.values()}
        digests = {name: _head_digest(folder / name) for name in cards.values()}
        _network_forbidden(monkeypatch)  # the size HEAD request fails: reported as "n/a"
        script = _load_script("fetch_weights")
        assert script.main(["--cards", "pixelseal", "videoseal_1.0", "videoseal_0.0"]) == 0
        out = capsys.readouterr().out
        assert "pixelseal: 256 bits" in out and "videoseal_0.0: 96 bits" in out
        assert {name: (folder / name).stat().st_mtime_ns for name in cards.values()} == before
        assert {name: _head_digest(folder / name) for name in cards.values()} == digests
        assert not list(folder.glob("*.part")) and not list(folder.glob("*.tmp"))


def _head_digest(path: Path, n_bytes: int = 1 << 20) -> str:
    with path.open("rb") as handle:
        return hashlib.sha256(handle.read(n_bytes)).hexdigest()
