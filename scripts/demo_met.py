"""Demo: Model Equality Testing (acceptance criterion 1).

1. Type-I error on a toy law (300 simulated H0 trials, rejection rate at alpha = 0.05).
2. Agreement of `mmd_hamming` with the official `model-equality-testing` package.
3. Power curves with SmolLM2-135M-Instruct: reference (T = 1) versus
   - the same model (H0, size),
   - temperature 0.7,
   - KGW watermark delta = 2.5 and delta = 1.0 (gamma = 0.25),
   - Gumbel-max watermark with a fixed key (deterministic: identical completions).
   Budgets are samples PER PROMPT; 10 prompts, L = 50 tokens, 100 trials per cell.

Pools of completions are cached in results/textmet/ so a re-run is cheap.
Heavy job: run it through `jarvis-heavy`.

    PYTHONPATH=$PWD/src jarvis-heavy env python scripts/demo_met.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import beta

from wmkit.audit import (
    METResult,
    Sample,
    mmd_hamming,
    power_curve,
    two_sample_test,
)
from wmkit.config import results_dir
from wmkit.text import (
    DEFAULT_LM,
    GumbelConfig,
    KGWConfig,
    generate_many,
    load_lm,
)

COMPLETION_LEN = 50
N_PROMPTS_TOY = 5
TOY_VOCAB = 12
TOY_LENGTH = 8
TOY_PER_SIDE = 40
TOY_TRIALS = 300
TOY_RESAMPLES = 199
TOY_SEED = 20261121
ALPHA = 0.05
CONFIDENCE = 0.95

POOL_PER_PROMPT = 60
POOL_SEEDS = {
    "ref_a": 100,
    "ref_b": 200,
    "temp_0.7": 300,
    "kgw_d2.5": 400,
    "kgw_d1.0": 500,
    "gumbel": 600,
}
BUDGETS_PER_PROMPT = (1, 2, 4, 8, 16, 32)
POWER_TRIALS = 100
POWER_RESAMPLES = 99
POWER_SEED = 5
FULL_TEST_RESAMPLES = 999
SIMULATION_PER_PROMPT = 20
KGW_GAMMA = 0.25
KGW_DELTA_PAPER = 2.5
KGW_DELTA_WEAK = 1.0
ALT_TEMPERATURE = 0.7

PROMPTS = (
    "The history of the Roman Empire began when",
    "In the heart of the forest, a small village",
    "La revolution francaise a commence",
    "To install the package, first you need to",
    "Scientists announced yesterday that a new",
    "Der Bahnhof von Berlin ist",
    "Once upon a time, in a distant kingdom,",
    "The quarterly report shows that revenue",
    "Photosynthesis is the process by which plants",
    "El presidente anuncio que el nuevo",
)

Pool = dict[int, list[tuple[int, ...]]]
"""Completions per prompt index."""


def format_table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a GitHub-flavoured markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def clopper_pearson(successes: int, trials: int) -> tuple[float, float]:
    """Clopper-Pearson interval of a proportion."""
    alpha = 1.0 - CONFIDENCE
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    high = (
        1.0
        if successes == trials
        else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    )
    return low, high


def toy_size_experiment() -> dict[str, Any]:
    """Part 1: rejection rate of the permutation test when both samples share one law."""
    rng = np.random.default_rng(TOY_SEED)
    law = rng.dirichlet(np.full(TOY_VOCAB, 0.7), size=(N_PROMPTS_TOY, TOY_LENGTH))

    def draw(n_samples: int) -> list[Sample]:
        out = []
        for index in range(n_samples):
            prompt = index % N_PROMPTS_TOY
            tokens = tuple(int(rng.choice(TOY_VOCAB, p=law[prompt, k])) for k in range(TOY_LENGTH))
            out.append(Sample(prompt, tokens))
        return out

    started = time.perf_counter()
    p_values = np.array(
        [
            two_sample_test(
                draw(TOY_PER_SIDE),
                draw(TOY_PER_SIDE),
                completion_len=TOY_LENGTH,
                n_resamples=TOY_RESAMPLES,
                seed=trial,
            ).p_value
            for trial in range(TOY_TRIALS)
        ]
    )
    rejections = int((p_values <= ALPHA).sum())
    low, high = clopper_pearson(rejections, TOY_TRIALS)
    deciles = np.histogram(p_values, bins=10, range=(0.0, 1.0))[0]
    print("## Criterion 1a: type-I error on a toy law (H0 simulated)\n")
    print(
        format_table(
            [
                "trials",
                "n per side",
                "B",
                "rejection rate at 0.05 [95% CI]",
                "accepted band",
                "mean p",
                "p histogram (deciles)",
                "seconds",
            ],
            [
                [
                    TOY_TRIALS,
                    TOY_PER_SIDE,
                    TOY_RESAMPLES,
                    f"{rejections}/{TOY_TRIALS} = {rejections / TOY_TRIALS:.3f} [{low:.3f}, {high:.3f}]",
                    "[0.02, 0.09]",
                    f"{p_values.mean():.3f}",
                    deciles.tolist(),
                    f"{time.perf_counter() - started:.1f}",
                ]
            ],
        )
    )
    return {"rejection_rate": rejections / TOY_TRIALS, "trials": TOY_TRIALS}


def official_agreement(
    samples_p: Sequence[Sample], samples_q: Sequence[Sample], n_prompts: int, completion_len: int
) -> tuple[float, float | None]:
    """Return (our statistic, official statistic or None when the package is missing)."""
    ours = mmd_hamming(samples_p, samples_q, completion_len=completion_len)
    try:
        from model_equality_testing.algorithm import run_two_sample_test
        from model_equality_testing.distribution import CompletionSample
        from model_equality_testing.utils import pad_to_length
    except ImportError:
        return ours, None

    def convert(samples: Sequence[Sample]) -> Any:
        completions = np.full((len(samples), completion_len), -1, dtype=np.int64)
        for row, sample in enumerate(samples):
            kept = sample.completion_ids[:completion_len]
            completions[row, : len(kept)] = kept
        return CompletionSample(
            prompts=np.array([s.prompt_id for s in samples]),
            completions=pad_to_length(completions, L=completion_len),
            m=n_prompts,
        )

    _, official = run_two_sample_test(
        convert(samples_p), convert(samples_q), pvalue_type="dummy", stat_type="mmd_hamming"
    )
    return ours, float(official)


def build_pools(model: Any, tokenizer: Any) -> tuple[dict[str, Pool], dict[str, float]]:
    """Sample every pool (POOL_PER_PROMPT completions per prompt), timing each."""
    specs: dict[str, dict[str, Any]] = {
        "ref_a": {"scheme": "none", "config": None, "temperature": 1.0},
        "ref_b": {"scheme": "none", "config": None, "temperature": 1.0},
        "temp_0.7": {"scheme": "none", "config": None, "temperature": ALT_TEMPERATURE},
        "kgw_d2.5": {
            "scheme": "kgw",
            "config": KGWConfig(gamma=KGW_GAMMA, delta=KGW_DELTA_PAPER),
            "temperature": 1.0,
        },
        "kgw_d1.0": {
            "scheme": "kgw",
            "config": KGWConfig(gamma=KGW_GAMMA, delta=KGW_DELTA_WEAK),
            "temperature": 1.0,
        },
        "gumbel": {"scheme": "gumbel", "config": GumbelConfig(), "temperature": 1.0},
    }
    pools: dict[str, Pool] = {}
    seconds: dict[str, float] = {}
    for name, spec in specs.items():
        started = time.perf_counter()
        pool: Pool = {}
        for index, prompt in enumerate(PROMPTS):
            # Gumbel is deterministic given key and prefix: one run, replicated.
            n_sequences = 1 if spec["scheme"] == "gumbel" else POOL_PER_PROMPT
            rows = generate_many(
                model,
                tokenizer,
                prompt,
                n_sequences,
                scheme=spec["scheme"],
                config=spec["config"],
                max_new_tokens=COMPLETION_LEN,
                seed=POOL_SEEDS[name] + index,
                temperature=spec["temperature"],
            )
            pool[index] = [tuple(row) for row in rows] * (POOL_PER_PROMPT // n_sequences)
        pools[name] = pool
        seconds[name] = time.perf_counter() - started
        print(
            f"  pool {name}: {len(PROMPTS) * POOL_PER_PROMPT} completions in {seconds[name]:.0f}s"
        )
    return pools, seconds


def load_pools(
    model_name: str, device: str | None
) -> tuple[dict[str, Pool], dict[str, float], int]:
    """Return cached pools (generating them on first use) and the parameter count."""
    fingerprint = json.dumps(
        [model_name, COMPLETION_LEN, POOL_PER_PROMPT, POOL_SEEDS, list(PROMPTS)]
    )
    cache_file = (
        results_dir()
        / "textmet"
        / f"met_pools_{hashlib.sha256(fingerprint.encode()).hexdigest()[:10]}.json"
    )
    model, tokenizer = load_lm(model_name, device)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    if cache_file.exists():
        print(f"using cached pools {cache_file}")
        raw = json.loads(cache_file.read_text())
        pools = {
            name: {int(j): [tuple(row) for row in rows] for j, rows in pool.items()}
            for name, pool in raw["pools"].items()
        }
        return pools, raw["seconds"], n_params
    print(f"sampling pools on {next(model.parameters()).device}")
    pools, seconds = build_pools(model, tokenizer)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(
            {
                "pools": {n: {str(j): rows for j, rows in p.items()} for n, p in pools.items()},
                "seconds": seconds,
            }
        )
    )
    return pools, seconds, n_params


def draw_from_pool(pool: Pool, per_prompt: int, rng: np.random.Generator) -> list[Sample]:
    """Draw `per_prompt` completions per prompt, without replacement."""
    samples = []
    for prompt_index, rows in pool.items():
        for choice in rng.choice(len(rows), size=per_prompt, replace=False):
            samples.append(Sample(prompt_index, rows[int(choice)]))
    return samples


def full_pool_samples(pool: Pool, per_prompt: int) -> list[Sample]:
    """First `per_prompt` completions of each prompt."""
    return [Sample(j, row) for j, rows in pool.items() for row in rows[:per_prompt]]


def power_experiment(pools: dict[str, Pool]) -> dict[str, Any]:
    """Part 3: power curves of the permutation test with the real model."""
    reference = pools["ref_a"]
    alternatives = {
        "H0: same model (ref_b), size": "ref_b",
        "temperature 0.7 vs 1.0": "temp_0.7",
        "KGW delta=2.5 (paper)": "kgw_d2.5",
        "KGW delta=1.0": "kgw_d1.0",
        "Gumbel-max, fixed key": "gumbel",
    }
    rows: list[list[object]] = []
    record: dict[str, Any] = {}
    for label, name in alternatives.items():
        started = time.perf_counter()
        alternative = pools[name]
        curve = power_curve(
            lambda n, rng: draw_from_pool(reference, n, rng),
            lambda n, rng, alt=alternative: draw_from_pool(alt, n, rng),  # type: ignore[misc]
            BUDGETS_PER_PROMPT,
            n_trials=POWER_TRIALS,
            alpha=ALPHA,
            seed=POWER_SEED,
            completion_len=COMPLETION_LEN,
            n_resamples=POWER_RESAMPLES,
        )
        rows.append(
            [label, *[f"{rate:.2f}" for _, rate in curve], f"{time.perf_counter() - started:.0f}"]
        )
        record[label] = {str(budget): rate for budget, rate in curve}
    header = [
        "alternative \\ samples per prompt (N = 10 x n per side)",
        *[str(b) for b in BUDGETS_PER_PROMPT],
        "seconds",
    ]
    print(
        f"\n## Criterion 1b: rejection rate at alpha = {ALPHA} ({POWER_TRIALS} trials, B = {POWER_RESAMPLES} permutations, 10 prompts, L = {COMPLETION_LEN})\n"
    )
    print(format_table(header, rows))
    print(
        "\nResampling is done from finite pools of 60 completions per prompt: trials are not fully independent for the largest budgets."
    )
    return record


def full_tests(pools: dict[str, Pool]) -> dict[str, Any]:
    """Single tests at 20 samples per prompt: permutation, simulation (Algorithm 1) and official statistic."""
    reference = full_pool_samples(pools["ref_a"], SIMULATION_PER_PROMPT)
    rows: list[list[object]] = []
    record: dict[str, Any] = {}
    for name in ("ref_b", "temp_0.7", "kgw_d2.5", "kgw_d1.0", "gumbel"):
        other = full_pool_samples(pools[name], SIMULATION_PER_PROMPT)
        started = time.perf_counter()
        permutation: METResult = two_sample_test(
            reference, other, completion_len=COMPLETION_LEN, n_resamples=FULL_TEST_RESAMPLES, seed=1
        )
        # Rows 20.. of ref_b: disjoint from every sample under test (ref_b[:20] is the H0 row).
        pool_for_null = [
            Sample(j, row)
            for j, rows_j in pools["ref_b"].items()
            for row in rows_j[SIMULATION_PER_PROMPT:]
        ]
        simulation = two_sample_test(
            reference,
            other,
            completion_len=COMPLETION_LEN,
            pvalue="simulation",
            reference_pool=pool_for_null,
            n_resamples=FULL_TEST_RESAMPLES,
            seed=1,
        )
        ours, official = official_agreement(reference, other, len(PROMPTS), COMPLETION_LEN)
        agreement = (
            "official package missing" if official is None else f"{abs(ours - official):.1e}"
        )
        rows.append(
            [
                name,
                f"{permutation.statistic:.5f}",
                f"{permutation.p_value:.4f}",
                f"{simulation.p_value:.4f}",
                agreement,
                f"{time.perf_counter() - started:.1f}",
            ]
        )
        record[name] = {
            "statistic": permutation.statistic,
            "p_permutation": permutation.p_value,
            "p_simulation": simulation.p_value,
            "abs_diff_vs_official": None if official is None else abs(ours - official),
        }
    print(
        f"\n## Single tests: {SIMULATION_PER_PROMPT} samples per prompt per side (N = {len(reference)}), B = {FULL_TEST_RESAMPLES}\n"
    )
    print(
        format_table(
            [
                "Q versus P = ref_a",
                "MMD hamming",
                "p (permutation)",
                "p (simulation, pool = ref_b[20:])",
                "|ours - official| statistic",
                "seconds",
            ],
            rows,
        )
    )
    return record


def toy_official_agreement() -> dict[str, float | None]:
    """Part 2: statistic agreement on several random toy datasets."""
    worst: float | None = None
    for seed in range(5):
        rng = np.random.default_rng(seed)
        laws = [
            rng.dirichlet(np.full(TOY_VOCAB, 0.7), size=(N_PROMPTS_TOY, TOY_LENGTH))
            for _ in range(2)
        ]
        sides = [
            [
                Sample(
                    i % N_PROMPTS_TOY,
                    tuple(
                        int(rng.choice(TOY_VOCAB, p=law[i % N_PROMPTS_TOY, k]))
                        for k in range(TOY_LENGTH)
                    ),
                )
                for i in range(60)
            ]
            for law in laws
        ]
        ours, official = official_agreement(sides[0], sides[1], N_PROMPTS_TOY, TOY_LENGTH)
        if official is not None:
            worst = max(worst or 0.0, abs(ours - official))
    print("\n## Criterion 1c: agreement with the official package (`stat_type='mmd_hamming'`)\n")
    if worst is None:
        print("model-equality-testing is not installed: concordance not checked.")
    else:
        print(f"5 random toy datasets (60 samples per side): max |ours - official| = {worst:.2e}")
    return {"max_abs_diff": worst}


def main() -> None:
    """Run the three parts and write results/textmet/met_demo.json."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", default=DEFAULT_LM)
    parser.add_argument("--device", default=None, help="torch device (default: cuda, mps, cpu)")
    parser.add_argument("--skip-model", action="store_true", help="toy parts only")
    args = parser.parse_args()

    started = time.perf_counter()
    record: dict[str, Any] = {
        "toy_size": toy_size_experiment(),
        "toy_official": toy_official_agreement(),
    }
    if not args.skip_model:
        pools, seconds, n_params = load_pools(args.model, args.device)
        print(
            f"\nmodel {args.model}: {n_params / 1e6:.1f} M parameters ({n_params * 4 / 1e6:.0f} MB in fp32); pool sampling seconds { ({k: round(v) for k, v in seconds.items()}) }"
        )
        record["power"] = power_experiment(pools)
        record["single_tests"] = full_tests(pools)
    out_file = results_dir() / "textmet" / "met_demo.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(record, indent=2))
    print(f"\nresults written to {out_file}; total {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
