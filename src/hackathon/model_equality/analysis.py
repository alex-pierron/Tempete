"""Offline analysis: stored answers -> channel statistics -> distances -> two-level clusters.

Nothing here touches the network. Typical use:

    calibration = calibrate_on_reference(ref_answers, truth)        # weights + thresholds
    analysis = run_analysis(ep_answers, ENDPOINTS, calibration=calibration)
    write_outputs(analysis, out_dir, tag="ep")

Row accounting is explicit at every step (rows read, rows used, rows rejected with their reason),
because a silently dropped answer is a silently wasted piece of the shared budget.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from hackathon.model_equality.boilerplate import BoilerplateNote, strip_boilerplate
from hackathon.model_equality.clustering import (
    ChannelDiagnostic,
    ThresholdCalibration,
    TwoLevelClustering,
    ambiguous_pairs,
    build_linkage,
    calibrate_model_threshold,
    calibrate_weights,
    combine_channels,
    cut_two_levels,
    family_floor_of,
    prior_weights,
    to_distance,
)
from hackathon.model_equality.features import answer_key, build_views, first_word, tokenize_words
from hackathon.model_equality.pairwise import (
    DEFAULT_PERMUTATIONS,
    KERNELS,
    Channel,
    all_pair_stats,
    gram_matrix,
)
from hackathon.model_equality.prompts import category_of
from hackathon.model_equality.prompts.bank import (
    CATEGORY_ARITH,
    CATEGORY_COMPLETION,
    CATEGORY_FORMAT,
)
from hackathon.model_equality.scoring import confusion
from hackathon.model_equality.store import Answer

FloatMatrix = npt.NDArray[np.float64]
IntVector = npt.NDArray[np.int64]

MIN_ANSWERS_PER_ENDPOINT = 10
"""Below this many answers an endpoint is flagged: its row of distances is not trustworthy."""
EXACT_MATCH_CATEGORIES: tuple[str, ...] = (CATEGORY_ARITH, CATEGORY_FORMAT, CATEGORY_COMPLETION)
"""Categories whose answers are near-deterministic: exact agreement is a meaningful rate."""
TOP_FIRST_WORDS = 3
TOP_HEIGHTS_SHOWN = 15
REASON_NOT_SELECTED = "endpoint_not_selected"
REASON_OUTSIDE_UNITS = "outside_selected_units"
CALIBRATION_FILENAME = "calibration.json"
FAMILY_DISTANCE_PREFIX = "distance_family_"
DEFAULT_TAU_MODEL = 4.0
"""Uncalibrated model cut (in combined-z units): a same-model pair scores ~N(0, 1) under H0, and
disguises inflate it, so 4 is a deliberately cautious stand-in until `ref-*` calibration exists."""
DECIMALS = 3
UnitKey = tuple[str, int]
"""`(prompt id, repeat)`."""


@dataclass(frozen=True)
class Calibration:
    """Weights and thresholds learned on endpoints of known truth."""

    weights: dict[Channel, float]
    tau_model: float
    family_floor: float
    mcc_model_in_sample: float

    def to_json(self) -> str:
        """Serialise (channels become 'category/kernel' keys)."""
        return json.dumps({
            "weights": {f"{c}/{k}": w for (c, k), w in self.weights.items()},
            "tau_model": self.tau_model,
            "family_floor": self.family_floor,
            "mcc_model_in_sample": self.mcc_model_in_sample,
        }, indent=2)

    @staticmethod
    def from_json(text: str) -> Calibration:
        """Inverse of `to_json`."""
        raw = json.loads(text)
        weights: dict[Channel, float] = {}
        for name, weight in raw["weights"].items():
            category, kernel = name.split("/", 1)
            weights[(category, kernel)] = float(weight)
        return Calibration(weights, float(raw["tau_model"]), float(raw["family_floor"]),
                           float(raw["mcc_model_in_sample"]))


@dataclass
class RowAccounting:
    """Rows in / used / rejected, so every answer is accounted for."""

    rows_in: int = 0
    rows_used: int = 0
    rejected: list[tuple[str, Answer]] = field(default_factory=list)
    empty_responses: int = 0

    def reason_counts(self) -> Counter[str]:
        """Number of rejected rows per reason."""
        return Counter(reason for reason, _ in self.rejected)


@dataclass
class Analysis:
    """Everything the analysis computed."""

    endpoints: list[str]
    accounting: RowAccounting
    answers_per_endpoint: dict[str, int]
    z_by_channel: dict[Channel, FloatMatrix]
    weights: dict[Channel, float]
    combined: FloatMatrix
    distance: FloatMatrix
    family_distance: FloatMatrix
    undefined_pairs: list[tuple[int, int]]
    clustering: TwoLevelClustering
    ambiguous: list[tuple[int, int, float]]
    exact_match: FloatMatrix
    diagnostics: list[ChannelDiagnostic] = field(default_factory=list)
    boilerplate: list[BoilerplateNote] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def select_answers(
    answers: Sequence[Answer],
    endpoints: Collection[str],
    units: Collection[UnitKey] | None = None,
) -> tuple[list[Answer], RowAccounting]:
    """Keep the answers of the wanted endpoints (and optionally of given phase units).

    Args:
        answers: Rows read from the stores.
        endpoints: Endpoints to analyse.
        units: When given, only these `(prompt id, repeat)` units are kept (to analyse a store at
            the sample shape of a phase, e.g. to calibrate `ref-*` like `ep-1`).
    """
    accounting = RowAccounting(rows_in=len(answers))
    kept: list[Answer] = []
    wanted = set(endpoints)
    for answer in answers:
        if answer.endpoint not in wanted:
            accounting.rejected.append((REASON_NOT_SELECTED, answer))
        elif units is not None and (answer.prompt_id, answer.repeat) not in units:
            accounting.rejected.append((REASON_OUTSIDE_UNITS, answer))
        else:
            kept.append(answer)
            accounting.empty_responses += int(not answer.response.strip())
    accounting.rows_used = len(kept)
    return kept, accounting


def compute_channels(
    answers: Sequence[Answer], endpoints: Sequence[str], n_permutations: int = DEFAULT_PERMUTATIONS
) -> dict[Channel, FloatMatrix]:
    """z-score matrix of every (category, kernel) channel that is defined for some pair."""
    index_of = {endpoint: position for position, endpoint in enumerate(endpoints)}
    by_category: dict[str, list[Answer]] = defaultdict(list)
    for answer in answers:
        by_category[category_of(answer.prompt_id)].append(answer)
    channels: dict[Channel, FloatMatrix] = {}
    for category, rows in sorted(by_category.items()):
        rows = sorted(rows, key=lambda a: (a.prompt_id, a.endpoint, a.repeat))
        prompt_ids = sorted({a.prompt_id for a in rows})
        prompt_position = {pid: n for n, pid in enumerate(prompt_ids)}
        prompt_index = np.array([prompt_position[a.prompt_id] for a in rows], dtype=np.int64)
        endpoint_of_row = np.array([index_of[a.endpoint] for a in rows], dtype=np.int64)
        rows_by_endpoint = {
            int(e): np.flatnonzero(endpoint_of_row == e) for e in np.unique(endpoint_of_row)
        }
        views = build_views(category, [a.response for a in rows])
        for kernel in KERNELS:
            gram = gram_matrix(kernel, views, prompt_index)
            z_matrix, _ = all_pair_stats(gram, prompt_index, rows_by_endpoint, len(endpoints),
                                         n_permutations)
            if np.isfinite(z_matrix).any():
                channels[(category, kernel)] = z_matrix
    return channels


def exact_match_matrix(answers: Sequence[Answer], endpoints: Sequence[str]) -> FloatMatrix:
    """Mean exact agreement of answer keys on near-deterministic prompts, per endpoint pair.

    For each prompt both endpoints answered, the fraction of equal-key answer pairs; then the
    mean over prompts. The diagonal is the within-endpoint agreement (a determinism measure:
    1.0 means the endpoint always gives the same answer). NaN where no prompt is shared.
    """
    index_of = {endpoint: position for position, endpoint in enumerate(endpoints)}
    keys: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for answer in answers:
        category = category_of(answer.prompt_id)
        if category in EXACT_MATCH_CATEGORIES and answer.endpoint in index_of:
            keys[answer.prompt_id][index_of[answer.endpoint]].append(
                answer_key(category, answer.response))
    size = len(endpoints)
    totals = np.zeros((size, size))
    counts = np.zeros((size, size))
    for per_endpoint in keys.values():
        for first, first_keys in per_endpoint.items():
            for second, second_keys in per_endpoint.items():
                if second < first:
                    continue
                if first == second:
                    pairs = [(x, y) for i, x in enumerate(first_keys)
                             for j, y in enumerate(first_keys) if i < j]
                else:
                    pairs = [(x, y) for x in first_keys for y in second_keys]
                if not pairs:
                    continue
                rate = sum(x == y for x, y in pairs) / len(pairs)
                for a, b in {(first, second), (second, first)}:
                    totals[a, b] += rate
                    counts[a, b] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)


def run_analysis(
    answers: Sequence[Answer],
    endpoints: Sequence[str],
    *,
    calibration: Calibration | None = None,
    weights: Mapping[Channel, float] | None = None,
    tau_model: float | None = None,
    tau_family: float | None = None,
    units: Collection[UnitKey] | None = None,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    strip: bool = True,
) -> Analysis:
    """Full offline analysis of `answers` over `endpoints`.

    Weights and thresholds come, in decreasing priority, from the explicit arguments, from
    `calibration`, or from the priors. Without any threshold information the model cut falls
    back to `DEFAULT_TAU_MODEL` and is flagged as uncalibrated by the report.

    Raises:
        ClusteringError: Nothing to cluster (no defined channel for any pair).
    """
    kept, accounting = select_answers(answers, endpoints, units)
    per_endpoint = Counter(answer.endpoint for answer in kept)
    cleaned, notes = strip_boilerplate(kept) if strip else (kept, [])
    z_by_channel = compute_channels(cleaned, endpoints, n_permutations)
    chosen_weights = dict(weights) if weights is not None else (
        dict(calibration.weights) if calibration is not None else prior_weights(list(z_by_channel)))
    for channel in z_by_channel:
        chosen_weights.setdefault(channel, prior_weights([channel])[channel])
    warnings: list[str] = []
    if not any(chosen_weights[channel] > 0 for channel in z_by_channel):
        warnings.append("every available channel has zero calibrated weight (the queried "
                        "categories differ from the calibrated ones): prior weights used")
        chosen_weights = prior_weights(list(z_by_channel))
    combined = combine_channels(z_by_channel, chosen_weights)
    distance, undefined = to_distance(combined)
    family_distance, _ = to_distance(combine_channels(z_by_channel, prior_weights(list(z_by_channel))))
    tau = tau_model if tau_model is not None else (
        calibration.tau_model if calibration is not None else DEFAULT_TAU_MODEL)
    floor = calibration.family_floor if calibration is not None else 0.0
    clustering = cut_two_levels(distance, family_distance, tau, tau_family, floor)
    return Analysis(
        endpoints=list(endpoints), accounting=accounting,
        answers_per_endpoint={e: per_endpoint.get(e, 0) for e in endpoints},
        z_by_channel=z_by_channel, weights=chosen_weights, combined=combined, distance=distance,
        family_distance=family_distance, undefined_pairs=undefined, clustering=clustering,
        ambiguous=ambiguous_pairs(combined, clustering),
        exact_match=exact_match_matrix(cleaned, endpoints), boilerplate=notes, warnings=warnings,
    )


def model_mcc(analysis: Analysis, labels: Sequence[tuple[str, str]]) -> float:
    """Model MCC of an analysis' model clusters against known `(family, model)` labels."""
    size = len(labels)
    pairs = [(i, j) for i in range(size) for j in range(i + 1, size)
             if labels[i][0] == labels[j][0]]
    truth = [labels[i] == labels[j] for i, j in pairs]
    ids = analysis.clustering.model_ids
    guess = [ids[i] == ids[j] for i, j in pairs]
    return confusion(truth, guess).mcc


def calibrate_on_reference(
    answers: Sequence[Answer],
    truth: Mapping[str, tuple[str, str, str]],
    *,
    units: Collection[UnitKey] | None = None,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    strip: bool = True,
) -> tuple[Calibration, list[ChannelDiagnostic], Analysis]:
    """Learn weights and the model threshold from the `ref-*` endpoints and their known truth.

    Returns:
        The calibration, the per-channel diagnostics, and the in-sample analysis of the
        reference endpoints (its model MCC is optimistic: the same data chose the weights).
    """
    endpoints = sorted(truth)
    labels = [(truth[e][0], truth[e][1]) for e in endpoints]
    kept, _ = select_answers(answers, endpoints, units)
    cleaned = strip_boilerplate(kept)[0] if strip else kept
    z_by_channel = compute_channels(cleaned, endpoints, n_permutations)
    weights, diagnostics = calibrate_weights(z_by_channel, labels)
    combined = combine_channels(z_by_channel, weights)
    distance, _ = to_distance(combined)
    threshold: ThresholdCalibration = calibrate_model_threshold(distance, labels)
    family_distance, _ = to_distance(
        combine_channels(z_by_channel, prior_weights(list(z_by_channel))))
    calibration = Calibration(weights, threshold.tau_model, family_floor_of(family_distance),
                              threshold.mcc_model)
    analysis = run_analysis(answers, endpoints, calibration=calibration, units=units,
                            n_permutations=n_permutations, strip=strip)
    analysis.diagnostics = diagnostics
    return calibration, diagnostics, analysis


def family_names(clustering: TwoLevelClustering) -> tuple[list[str], list[str]]:
    """Readable labels: families A, B, ... by decreasing size; models A1, A2, ... inside each."""
    family_sizes = Counter(clustering.family_ids)
    ordered_families = sorted(family_sizes, key=lambda f: (-family_sizes[f], f))
    letter = {family: _letters(rank) for rank, family in enumerate(ordered_families)}
    model_sizes = Counter(clustering.model_ids)
    model_label: dict[int, str] = {}
    next_index: Counter[str] = Counter()
    for model in sorted(model_sizes, key=lambda m: (-model_sizes[m], m)):
        owner = next(clustering.family_ids[i] for i, m in enumerate(clustering.model_ids) if m == model)
        next_index[letter[owner]] += 1
        model_label[model] = f"{letter[owner]}{next_index[letter[owner]]}"
    return ([letter[f] for f in clustering.family_ids],
            [model_label[m] for m in clustering.model_ids])


def _letters(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA (the official template's family labels)."""
    name, number = "", index + 1
    while number:
        number, remainder = divmod(number - 1, 26)
        name = chr(65 + remainder) + name
    return name


def fingerprints(answers: Sequence[Answer], endpoints: Sequence[str]) -> list[str]:
    """One line per endpoint: answer count, mean length in words, empty replies, top first words."""
    lines = []
    for endpoint in endpoints:
        mine = [a for a in answers if a.endpoint == endpoint]
        if not mine:
            lines.append(f"{endpoint}: no answers")
            continue
        lengths = [len(tokenize_words(a.response)) for a in mine]
        firsts = Counter(first_word(a.response) or "<empty>" for a in mine).most_common(TOP_FIRST_WORDS)
        lines.append(
            f"{endpoint}: n={len(mine)} mean_words={np.mean(lengths):.1f} "
            f"empty={sum(not a.response.strip() for a in mine)} "
            f"first_words={', '.join(f'{w}({c})' for w, c in firsts)}")
    return lines


def render_report(analysis: Analysis, answers: Sequence[Answer], tag: str) -> str:
    """Plain-text report: accounting, channels, thresholds, clusters, ambiguities, warnings."""
    a = analysis
    family_labels, model_labels = family_names(a.clustering)
    lines = [f"# model-equality analysis [{tag}]", ""]
    reasons = a.accounting.reason_counts()
    lines.append(f"rows in: {a.accounting.rows_in}  used: {a.accounting.rows_used}  "
                 f"rejected: {len(a.accounting.rejected)} "
                 f"({', '.join(f'{r}={n}' for r, n in sorted(reasons.items())) or 'none'})  "
                 f"empty responses kept: {a.accounting.empty_responses}")
    lines.extend(f"WARNING: {message}" for message in a.warnings)
    short = [e for e, n in a.answers_per_endpoint.items() if n < MIN_ANSWERS_PER_ENDPOINT]
    if short:
        lines.append(f"WARNING: fewer than {MIN_ANSWERS_PER_ENDPOINT} answers for: {', '.join(short)}")
    if a.undefined_pairs:
        lines.append(f"WARNING: {len(a.undefined_pairs)} pair(s) with no defined channel "
                     f"(filled with the mean distance): "
                     + ", ".join(f"{a.endpoints[i]}~{a.endpoints[j]}" for i, j in a.undefined_pairs[:10]))
    lines += ["", (f"model cut tau_model = {a.clustering.tau_model:.2f}, "
                  f"family cut tau_family = {a.clustering.tau_family:.2f} (combined-z units)"), ""]
    heights = sorted((float(h) for h in build_linkage(a.distance)[:, 2]), reverse=True)
    lines.append("top merge heights of the model dendrogram (look for a gap to place tau_model): "
                 + " ".join(f"{h:.1f}" for h in heights[:TOP_HEIGHTS_SHOWN]))
    lines.append("")
    lines.append("## channels (weight, calibration diagnostics when available)")
    diag = {d.channel: d for d in a.diagnostics}
    for channel in sorted(a.weights, key=lambda c: -a.weights[c]):
        extra = ""
        if channel in diag:
            d = diag[channel]
            extra = (f"  same={d.mean_same:.2f} diff={d.mean_different:.2f} auc={d.auc:.2f}")
        lines.append(f"  {channel[0]}/{channel[1]:<13} w={a.weights[channel]:.3f}{extra}")
    if a.boilerplate:
        lines += ["", "## boilerplate stripped (persona / system-prompt phrases)"]
        lines += [f"  {n.endpoint} {n.side}: '{n.phrase}' in {n.count}/{n.total} answers"
                  for n in a.boilerplate]
    lines += ["", "## clusters"]
    tree: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for endpoint, family, model in zip(a.endpoints, family_labels, model_labels, strict=True):
        tree[family][model].append(endpoint)
    for family in sorted(tree):
        lines.append(f"family {family}")
        for model in sorted(tree[family]):
            lines.append(f"  model {model}: {' '.join(tree[family][model])}")
    lines += ["", f"## ambiguous same-family pairs (|Z - tau| <= 30 %): {len(a.ambiguous)}"]
    pressure: Counter[str] = Counter()
    for i, j, z in a.ambiguous[:20]:
        lines.append(f"  {a.endpoints[i]} ~ {a.endpoints[j]}  Z={z:.2f}")
    for i, j, _ in a.ambiguous:
        pressure[a.endpoints[i]] += 1
        pressure[a.endpoints[j]] += 1
    if pressure:
        lines.append("  endpoints to feed first (phase 3): "
                     + ", ".join(f"{e}({n})" for e, n in pressure.most_common(10)))
    lines += ["", "## fingerprints"] + fingerprints(answers, a.endpoints)
    return "\n".join(lines) + "\n"


def write_matrix(path: Path, endpoints: Sequence[str], matrix: FloatMatrix) -> None:
    """Write a labelled square matrix as CSV (NaN as an empty cell)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", *endpoints])
        for endpoint, row in zip(endpoints, matrix, strict=True):
            writer.writerow([endpoint, *("" if np.isnan(v) else f"{v:.{DECIMALS}f}" for v in row)])


def read_matrix(path: Path) -> tuple[list[str], FloatMatrix]:
    """Inverse of `write_matrix`."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    endpoints = rows[0][1:]
    matrix = np.array([[float(v) if v else np.nan for v in row[1:]] for row in rows[1:]])
    return endpoints, matrix.astype(np.float64)


def write_outputs(analysis: Analysis, answers: Sequence[Answer], out_dir: Path, tag: str) -> list[Path]:
    """Write report, distance matrix, labels, exact-match matrix (and rejected rows if any)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    family_labels, model_labels = family_names(analysis.clustering)
    written = [out_dir / f"report_{tag}.txt", out_dir / f"distance_{tag}.csv",
               out_dir / f"labels_{tag}.csv", out_dir / f"exact_match_{tag}.csv",
               out_dir / f"{FAMILY_DISTANCE_PREFIX}{tag}.csv"]
    written[0].write_text(render_report(analysis, answers, tag), encoding="utf-8")
    write_matrix(written[1], analysis.endpoints, analysis.distance)
    with written[2].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["endpoint", "family", "model"])
        writer.writerows(zip(analysis.endpoints, family_labels, model_labels, strict=True))
    write_matrix(written[3], analysis.endpoints, analysis.exact_match)
    write_matrix(written[4], analysis.endpoints, analysis.family_distance)
    if analysis.accounting.rejected:
        rejected_path = out_dir / f"rejected_{tag}.jsonl"
        with rejected_path.open("w", encoding="utf-8") as handle:
            for reason, answer in analysis.accounting.rejected:
                handle.write(json.dumps({"reason": reason, "endpoint": answer.endpoint,
                                         "id": answer.prompt_id, "repeat": answer.repeat}) + "\n")
        written.append(rejected_path)
    return written
