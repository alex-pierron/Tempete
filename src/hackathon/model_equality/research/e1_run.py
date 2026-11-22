"""Phase E1 end to end: families, models, ref check at N = 5, files.

Two levels, two distances (both fixed a priori, then checked on the ref):

* FAMILY distance (a different base model): per prompt, a categorical dissimilarity on the
  signals that differ between base models: the CREATOR named by the identity prompt (looked up
  in a lexicon; an unknown creator counts 0.5 against anyone, so noise never splits alone) and,
  for the two yes/no prompts, the polarity of the answer plus its written style (capital letter,
  final full stop), which is a strong signature of a base model. Average linkage, cut at the
  largest top-merge height seen on the ref, where there is exactly one base model.
* MODEL distance (same family, different weights): the calibrated small-N statistic
  `ham+len` (see `smalln.py`), average linkage inside each family, cut at the held-out MCC
  optimum measured on the ref at N = 5.

Nothing here touches the network.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import fcluster

from hackathon.model_equality.analysis import write_matrix
from hackathon.model_equality.planning import SCORED_ENDPOINTS
from hackathon.model_equality.prompts.phases import E1_PROMPT_IDS
from hackathon.model_equality.research.small_n_eval import (
    CHOSEN_METHOD,
    METHOD_WEIGHTS,
    REF_ENDPOINTS,
    load_ref_answers,
    ref_truth,
)
from hackathon.model_equality.scoring import confusion
from hackathon.model_equality.smalln import (
    SmallNData,
    build_small_n_data,
    build_tree,
    pair_distance,
)
from hackathon.model_equality.store import Answer, AnswerStore

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
E1_N_PROMPTS = len(E1_PROMPT_IDS)
IDENTITY_PROMPT_ID = "fam-identity-000"
YESNO_PROMPT_IDS: tuple[str, ...] = ("cand-yesno-002", "cand-yesno-005")
UNKNOWN_CREATOR_DISTANCE = 0.5
"""Distance when one side names no known creator: neither evidence for nor against."""
POLARITY_WEIGHT = 0.5
STYLE_WEIGHT = 0.5
CREATOR_LEXICON: dict[str, tuple[str, ...]] = {
    "alibaba": ("qwen", "alibaba", "tongyi"),
    "meta": ("meta", "llama"),
    "anthropic": ("anthropic", "claude"),
    "openai": ("openai", "chatgpt", "gpt"),
    "google": ("google", "gemma", "gemini", "deepmind"),
    "microsoft": ("microsoft", "phi"),
    "mistral": ("mistral",),
    "huggingface": ("hugging face", "huggingface", "smollm"),
    "deepseek": ("deepseek",),
}
YESNO_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
Responses = Mapping[tuple[str, str], str]
"""`(endpoint, prompt id) -> response` for one draw of each prompt."""


@dataclass(frozen=True)
class Labels:
    """Nested family and model ids of the endpoints (ids are dense integers)."""

    families: IntArray
    models: IntArray


def creator_of(text: str) -> str | None:
    """The first known creator named in the text (earliest mention), or None."""
    lowered = text.lower()
    best: tuple[int, str] | None = None
    for name, keywords in CREATOR_LEXICON.items():
        for keyword in keywords:
            match = re.search(r"\b" + re.escape(keyword) + r"\b", lowered)
            if match and (best is None or match.start() < best[0]):
                best = (match.start(), name)
    return None if best is None else best[1]


def style_signature(text: str) -> tuple[bool, bool]:
    """(starts with a capital, ends with a full stop): how the answer is written."""
    stripped = text.strip()
    return stripped[:1].isupper(), stripped.endswith(".")


def polarity_of(text: str) -> str | None:
    """'yes' or 'no' (first one found), or None."""
    match = YESNO_PATTERN.search(text)
    return None if match is None else match.group(1).lower()


def family_distance(
    responses: Responses, endpoints: Sequence[str], skip: frozenset[str] = frozenset(),
) -> FloatArray:
    """Pairwise family dissimilarity from the identity and yes/no prompts (see module docs).

    Args:
        responses: `(endpoint, prompt id) -> text`.
        endpoints: Endpoints in matrix order.
        skip: Prompt ids to leave out (leave-one-prompt-out sensitivity).
    """
    size = len(endpoints)
    use_identity = IDENTITY_PROMPT_ID not in skip
    creators = [creator_of(responses[(e, IDENTITY_PROMPT_ID)]) if use_identity else None
                for e in endpoints]
    matrix = np.zeros((size, size), dtype=np.float64)
    for i, j in itertools.combinations(range(size), 2):
        first, second = creators[i], creators[j]
        parts = ([UNKNOWN_CREATOR_DISTANCE if first is None or second is None
                  else float(first != second)] if use_identity else [])
        for prompt_id in (p for p in YESNO_PROMPT_IDS if p not in skip):
            text_i, text_j = responses[(endpoints[i], prompt_id)], responses[(endpoints[j], prompt_id)]
            parts.append(POLARITY_WEIGHT * float(polarity_of(text_i) != polarity_of(text_j))
                         + STYLE_WEIGHT * float(style_signature(text_i) != style_signature(text_j)))
        matrix[i, j] = matrix[j, i] = float(np.mean(parts))
    return matrix


def label_endpoints(
    family_dist: FloatArray, model_dist: FloatArray, tau_family: float, tau_model: float,
) -> Labels:
    """Families from the family tree, then models inside each family from the model tree."""
    families = np.asarray(
        fcluster(build_tree(family_dist), t=tau_family, criterion="distance") - 1, dtype=np.int64)
    models = np.zeros(len(families), dtype=np.int64)
    next_id = 0
    for family in np.unique(families):
        members = np.flatnonzero(families == family)
        if len(members) == 1:
            local = np.zeros(1, dtype=np.int64)
        else:
            sub = model_dist[np.ix_(members, members)]
            local = np.asarray(
                fcluster(build_tree(sub), t=tau_model, criterion="distance") - 1, dtype=np.int64)
        models[members] = local + next_id
        next_id += int(local.max()) + 1
    return Labels(families=families, models=models)


def _responses_of(answers: Sequence[Answer], draws: Mapping[str, int]) -> dict[tuple[str, str], str]:
    """Responses of the chosen draw of each E1 prompt."""
    return {(a.endpoint, a.prompt_id): a.response for a in answers
            if a.prompt_id in draws and a.repeat == draws[a.prompt_id]}


def _draw_combinations(answers: Sequence[Answer], endpoints: Sequence[str]) -> list[dict[str, int]]:
    """Every choice of one draw (0 or 1) per E1 prompt that all endpoints have answered."""
    have: dict[str, set[int]] = {pid: set() for pid in E1_PROMPT_IDS}
    count: dict[tuple[str, int], int] = {}
    for a in answers:
        if a.prompt_id in have and a.endpoint in endpoints:
            count[(a.prompt_id, a.repeat)] = count.get((a.prompt_id, a.repeat), 0) + 1
    for (pid, repeat), n in count.items():
        if n == len(endpoints):
            have[pid].add(repeat)
    choices = [sorted(have[pid]) for pid in E1_PROMPT_IDS]
    if any(not c for c in choices):
        return []
    return [dict(zip(E1_PROMPT_IDS, combo, strict=True)) for combo in itertools.product(*choices)]


def ref_family_threshold(ref_answers: Sequence[Answer]) -> tuple[float, list[float]]:
    """Largest top-merge height of the family tree over the ref draw combinations (one family)."""
    heights = []
    for draws in _draw_combinations(ref_answers, REF_ENDPOINTS):
        distance = family_distance(_responses_of(ref_answers, draws), REF_ENDPOINTS)
        heights.append(float(build_tree(distance)[-1, 2]))
    return max(heights), heights


def ref_check(
    ref_answers: Sequence[Answer], tau_model: float, tau_family: float, method: str,
) -> dict[str, Any]:
    """Run the full E1 pipeline on the ref for every available draw combination.

    The five prompts were chosen on these ref answers, so the figure is optimistic; the
    held-out figure comes from `small_n_eval`.
    """
    pairs, truth = ref_truth()
    rows: list[dict[str, Any]] = []
    for draws in _draw_combinations(ref_answers, REF_ENDPOINTS):
        units = {(pid, rep) for pid, rep in draws.items()}
        data = build_small_n_data(ref_answers, list(REF_ENDPOINTS), units)
        model_dist = pair_distance(data, list(range(len(data.units))), METHOD_WEIGHTS[method])
        family_dist = family_distance(_responses_of(ref_answers, draws), REF_ENDPOINTS)
        labels = label_endpoints(family_dist, model_dist, tau_family, tau_model)
        guess = [labels.models[i] == labels.models[j] for i, j in pairs]
        rows.append({"draws": draws, "mcc_model": confusion(truth, guess).mcc,
                     "n_families": len(set(labels.families.tolist())),
                     "n_models": len(set(labels.models.tolist()))})
    mccs = [r["mcc_model"] for r in rows]
    return {"combinations": len(rows), "mcc_mean": float(np.mean(mccs)),
            "mcc_min": float(np.min(mccs)), "mcc_max": float(np.max(mccs)), "rows": rows}


def _by_size(labels: IntArray) -> list[int]:
    """Cluster ids ordered by decreasing size (ties by id)."""
    ids, counts = np.unique(labels, return_counts=True)
    return [int(i) for i, _ in sorted(zip(ids, counts, strict=True), key=lambda x: (-x[1], x[0]))]


def _letters(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA (the official template's family labels)."""
    name, number = "", index + 1
    while number:
        number, remainder = divmod(number - 1, 26)
        name = chr(65 + remainder) + name
    return name


def readable_names(labels: Labels) -> tuple[list[str], list[str]]:
    """Family letters A, B, ... by decreasing size; models A1, A2, ... inside each family."""
    family_name = {f: _letters(rank) for rank, f in enumerate(_by_size(labels.families))}
    counters: dict[str, int] = {}
    model_name: dict[int, str] = {}
    for model in _by_size(labels.models):
        first = int(np.flatnonzero(labels.models == model)[0])
        owner = family_name[int(labels.families[first])]
        counters[owner] = counters.get(owner, 0) + 1
        model_name[model] = f"{owner}{counters[owner]}"
    return ([family_name[int(f)] for f in labels.families],
            [model_name[int(m)] for m in labels.models])


def write_answers_table(answers: Sequence[Answer], path: Path) -> None:
    """The E1 answers of every endpoint, one row per endpoint, to read the clusters by eye."""
    by_key = {(a.endpoint, a.prompt_id): a.response for a in answers if a.repeat == 0}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["endpoint", *E1_PROMPT_IDS])
        for endpoint in SCORED_ENDPOINTS:
            writer.writerow([endpoint, *(by_key.get((endpoint, pid), "").replace("\n", " ")[:300]
                                         for pid in E1_PROMPT_IDS)])


def ep_pipeline(
    ep_answers: Sequence[Answer], tau_model: float, tau_family: float, method: str,
) -> tuple[SmallNData, FloatArray, FloatArray, Labels]:
    """Distances and labels of the 33 ep from the E1 answers (draw 0 of each prompt)."""
    units = {(pid, 0) for pid in E1_PROMPT_IDS}
    data = build_small_n_data(ep_answers, list(SCORED_ENDPOINTS), units)
    model_dist = pair_distance(data, list(range(len(data.units))), METHOD_WEIGHTS[method])
    draws = {pid: 0 for pid in E1_PROMPT_IDS}
    family_dist = family_distance(_responses_of(ep_answers, draws), SCORED_ENDPOINTS)
    return data, family_dist, model_dist, label_endpoints(family_dist, model_dist, tau_family,
                                                           tau_model)


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", type=Path, default=Path("artifacts/model_equality"))
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method", default=CHOSEN_METHOD)
    parser.add_argument("--tau-model", type=float, default=None)
    parser.add_argument("--tau-family", type=float, default=None)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    row = next(r for r in calibration["rows"]
               if r["method"] == args.method and r["n_prompts"] == E1_N_PROMPTS)
    ref_answers = load_ref_answers(args.store_dir)
    tau_family_ref, family_heights = ref_family_threshold(ref_answers)
    tau_model = args.tau_model if args.tau_model is not None else row["tau_model"]
    tau_family = args.tau_family if args.tau_family is not None else tau_family_ref
    ref = ref_check(ref_answers, tau_model, tau_family, args.method)
    ep_answers = list(AnswerStore(args.store_dir / "answers.jsonl").load().values())
    data, family_dist, model_dist, labels = ep_pipeline(ep_answers, tau_model, tau_family,
                                                        args.method)
    families, models = readable_names(labels)
    args.out.mkdir(parents=True, exist_ok=True)
    write_matrix(args.out / "distance_ep.csv", list(data.endpoints), model_dist)
    write_matrix(args.out / "distance_family_ep.csv", list(data.endpoints), family_dist)
    with (args.out / "labels_ep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["endpoint", "family", "model"])
        writer.writerows(zip(data.endpoints, families, models, strict=True))
    write_answers_table(ep_answers, args.out / "e1_answers.csv")
    draws = {pid: 0 for pid in E1_PROMPT_IDS}
    creators = {e: creator_of(_responses_of(ep_answers, draws)[(e, IDENTITY_PROMPT_ID)])
                for e in SCORED_ENDPOINTS}
    summary = {
        "method": args.method, "tau_model": tau_model, "tau_family": tau_family,
        "ref_family_top_merge_heights": family_heights,
        "ep_family_top_merge_heights": build_tree(family_dist)[-8:, 2].round(3).tolist(),
        "n_families": len(set(families)), "n_models": len(set(models)),
        "ep_creators": creators, "ref_check": ref, "calibration_row": row,
    }
    (args.out / "e1_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"tau_model {tau_model:.3f} tau_family {tau_family:.3f} -> {summary['n_families']} "
          f"families, {summary['n_models']} models; ref E1 MCC mean {ref['mcc_mean']:+.3f} "
          f"[{ref['mcc_min']:+.2f}, {ref['mcc_max']:+.2f}] over {ref['combinations']} draws")


if __name__ == "__main__":
    main()
