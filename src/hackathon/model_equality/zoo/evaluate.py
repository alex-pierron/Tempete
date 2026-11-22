"""Frozen rules of teams C and D measured on the zoo (Mac side, numpy only, no network).

Nothing is tuned here: the band edge, the threshold, the lexicon, the number of identity units and
the likelihood limit are the values frozen by r-met-c (`preregistered*.json`, `c_rehearsal`,
`c_identity`) and r-met-d (`d_local_run`); the bootstrap protocol and its seed are theirs too.

Rules, all on the same replicate draws (paired):

* `v2`      band rule on the full-stop rate of N arithmetic answers (`c_rehearsal`);
* `v3`      v2 + identity flag, frozen lexicon, k identity units (`c_identity`);
* `v3ext`   v3 with `mistral` added to the lexicon (NOT frozen; sensitivity: the frozen lexicon has
            no Mistral, so every Mistral endpoint would read as disguised);
* `v4`      team C's frozen v4 (`preregistered_v4.json`, `c_v4`): band RELATIVE to the family (flag iff
            rate >= max(0.45, 25th percentile of the family's rates + 0.20)), identity override with
            k = 3 units (3 formulations x 1 draw), frozen lexicon; `v4ext` adds `mistral`;
* `lik`     v2 + likelihood flag: an endpoint whose per-token log-likelihood under the base model
            differs from its reference by |t| >= 3 is disguised (`d_local_run.DISGUISE_T_LIMIT`),
            reference = the plain endpoint of the family (`lik_plain`, labels known, as on the ref)
            or the median of the other endpoints (`lik_median`, label-free, what an ep run can do);
* `lik_median40` the median-reference flag on 40 shared cells instead of N (more answers: what the
  flag needs to see a system prompt or a temperature, at the price of more ep budget);
* `v3ext+lik_*` both flags (union);
* `oracle`  a perfect disguise detector (no S endpoint is ever isolated): the ceiling of the band.

Scopes: each family alone (one family: score = 0.9 x MCC_model) and the 4 families together (36
endpoints: 0.1 x MCC_family + 0.9 x MCC_model) with the true families given (`oracle`) or found by
the likelihood of the answers under the four base models (`likelihood`).
"""

from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from hackathon.model_equality.research.c_identity import BASE_PATTERN, BASE_TOKENS
from hackathon.model_equality.research.c_pairs import Partition
from hackathon.model_equality.research.c_rehearsal import (
    SEED,
    Rule,
    ends_with_full_stop,
    official_score,
)
from hackathon.model_equality.zoo.config import (
    BASE_SLOTS,
    FAMILIES,
    MODEL_MAIN,
    REPEATS_ARITHMETIC,
    VARIANT_BY_KEY,
    endpoint_name,
    variant_for_slot,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

BAND_LOW = 0.55
"""Frozen band edge of v2 (`preregistered.json`)."""
THRESHOLD_X = 0.40
"""Frozen spend threshold of the team-C chain."""
TEAM_DECISION_SCORE = 0.60
"""Team decision: no ep spending below this score, validated on new data."""
REPLICATES = 1000
IDENTITY_K = 6
"""Adoption setting of the frozen v3 (3 formulations x 2 draws)."""
LIKELIHOOD_T_LIMIT = 3.0
"""`d_local_run.DISGUISE_T_LIMIT`."""
V4_DELTA = 0.20
V4_FLOOR = 0.45
V4_BASELINE_PERCENTILE = 25.0
V4_IDENTITY_UNITS = 3
"""`preregistered_v4.json`: relative band edge, floor, baseline percentile, identity units."""
SAMPLE_SIZES: tuple[int, ...] = (10, 15, 20, 40)
LIKELIHOOD_LARGE_CELLS = 40
ARITHMETIC_PATTERN = re.compile(r"^(h-arith|p10-arith|z-arith)-\d{3}$")
IDENTITY_PATTERN = re.compile(r"^d-ident-\d{3}$")
EXTENDED_TOKENS: tuple[str, ...] = (*BASE_TOKENS, "mistral", "mistralai")
EXTENDED_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t).replace(r"\ ", r"\s+") for t in EXTENDED_TOKENS) + r")\b",
    re.IGNORECASE,
)
PERCENTILES: tuple[float, float] = (10.0, 90.0)
RULES: tuple[str, ...] = (
    "v2", "v3", "v3ext", "v4", "v4ext", "lik_plain", "lik_median", "lik_median40", "v3ext+lik_median",
    "v3ext+lik_median40", "oracle",
)
SCOPE_ORACLE = "oracle_families"
SCOPE_LIKELIHOOD = "likelihood_families"
SCOPE_SINGLE = "single_family"
EXPECTED_REPEATS = REPEATS_ARITHMETIC
W8A8_SUFFIX = ":w8a8"
FP8_MODES: tuple[str, ...] = ("fp8_w", "fp8_w8a8")


class ZooDataError(ValueError):
    """The zoo answers or likelihood files are missing, ragged or inconsistent."""


@dataclass(frozen=True)
class ZooData:
    """Every zoo answer as arrays, endpoints ordered family by family then by slot.

    Attributes:
        endpoints: endpoint names.
        family_index: true family of each endpoint (index into `families`).
        slot: ref slot of each endpoint (1..10).
        truth_model: globally unique true model id of each endpoint.
        families: family keys present.
        prompts: arithmetic prompt ids (sorted).
        stops: `[E, P, R]` 1.0 when the stripped answer ends with a full stop.
        ll_sum / ll_tokens: `[S, E, P, R]` log-likelihood and token count of every answer under
            scorer `s` (family `families[s]`'s base model); NaN where not scored.
        identity_marked: `[E, U]` identity answer names none of the frozen base-model names.
        identity_marked_ext: same with `mistral` in the lexicon.
    """

    endpoints: tuple[str, ...]
    family_index: IntArray
    slot: IntArray
    truth_model: IntArray
    families: tuple[str, ...]
    prompts: tuple[str, ...]
    stops: FloatArray
    ll_sum: FloatArray
    ll_tokens: FloatArray
    identity_marked: BoolArray
    identity_marked_ext: BoolArray

    @property
    def endpoint_count(self) -> int:
        """Number of endpoints."""
        return len(self.endpoints)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_likelihood_file(directory: Path, scorer: str) -> dict[str, Any] | None:
    """`ll_<scorer>.json` (or its `.json.gz` archive) as a dict, `None` when absent."""
    plain = directory / f"ll_{scorer}.json"
    packed = directory / f"ll_{scorer}.json.gz"
    if plain.is_file():
        loaded = json.loads(plain.read_text(encoding="utf-8"))
    elif packed.is_file():
        with gzip.open(packed, "rt", encoding="utf-8") as handle:
            loaded = json.load(handle)
    else:
        return None
    if not isinstance(loaded, dict):
        raise ZooDataError(f"likelihood file of {scorer} is not a JSON object")
    return loaded


def load_zoo(directory: Path, families: Sequence[str] | None = None,
             overlay: Path | None = None) -> ZooData:
    """Read `zoo_<family>.jsonl` and `ll_<family>.json` of the families present in `directory`.

    `overlay`: a second run directory (same file names) whose endpoints REPLACE those of the
    main one and whose likelihood entries are merged in; used to swap the LoRA endpoints for the
    ones of another adapter while keeping every other variant.

    Raises:
        ZooDataError: no family, a missing likelihood file, or ragged answer grids.
    """
    chosen = [f.key for f in FAMILIES if (families is None or f.key in families)
              and (directory / f"zoo_{f.key}.jsonl").is_file()]
    if not chosen:
        raise ZooDataError(f"no zoo_<family>.jsonl in {directory}")
    names: list[str] = []
    family_index: list[int] = []
    slots: list[int] = []
    truth: list[int] = []
    model_ids: dict[tuple[str, str], int] = {}
    answers: dict[str, dict[tuple[str, int], str]] = {}
    for f_index, key in enumerate(chosen):
        replaced: set[str] = set()
        sources = [directory / f"zoo_{key}.jsonl"]
        if overlay is not None and (overlay / f"zoo_{key}.jsonl").is_file():
            sources.append(overlay / f"zoo_{key}.jsonl")
        for source in sources:
            for row in _read_jsonl(source):
                endpoint = str(row["endpoint"])
                if source != sources[0] and endpoint not in replaced:
                    answers[endpoint] = {}
                    replaced.add(endpoint)
                answers.setdefault(endpoint, {})[(str(row["id"]), int(str(row["repeat"])))] = (
                    str(row["response"]))
        for slot in range(1, 11):
            name = endpoint_name(key, slot)
            if name not in answers:
                continue
            names.append(name)
            family_index.append(f_index)
            slots.append(slot)
            model = variant_for_slot(slot).truth_model
            truth.append(model_ids.setdefault((key, model), len(model_ids)))
    prompt_ids = sorted({pid for e in names for (pid, _) in answers[e]
                         if ARITHMETIC_PATTERN.match(pid)})
    repeats = EXPECTED_REPEATS
    count = len(names)
    stops = np.full((count, len(prompt_ids), repeats), np.nan)
    for i, name in enumerate(names):
        for (pid, rep), text in answers[name].items():
            if ARITHMETIC_PATTERN.match(pid) and rep < repeats:
                stops[i, prompt_ids.index(pid), rep] = float(ends_with_full_stop(text))
    if np.isnan(stops).any():
        raise ZooDataError("ragged arithmetic answers: some (endpoint, prompt, repeat) is missing")
    unit_keys = sorted({(pid, rep) for e in names for (pid, rep) in answers[e]
                        if IDENTITY_PATTERN.match(pid)})
    marked = np.zeros((count, len(unit_keys)), dtype=np.bool_)
    marked_ext = np.zeros_like(marked)
    for i, name in enumerate(names):
        for u, unit in enumerate(unit_keys):
            identity_text = answers[name].get(unit)
            if identity_text is None:
                raise ZooDataError(f"{name} has no identity answer {unit}")
            marked[i, u] = BASE_PATTERN.search(identity_text) is None
            marked_ext[i, u] = EXTENDED_PATTERN.search(identity_text) is None
    ll_sum = np.full((len(chosen), count, len(prompt_ids), repeats), np.nan)
    ll_tokens = np.full_like(ll_sum, np.nan)
    for s, scorer in enumerate(chosen):
        payload = read_likelihood_file(directory, scorer)
        if payload is None:
            continue
        base = dict(payload["scores"]["base"])
        extra = read_likelihood_file(overlay, scorer) if overlay is not None else None
        if extra is not None:
            base.update(extra["scores"]["base"])
        for i, name in enumerate(names):
            for p, pid in enumerate(prompt_ids):
                for rep in range(repeats):
                    entry = base.get(f"{name}|{pid}|{rep}")
                    if entry is not None:
                        ll_sum[s, i, p, rep], ll_tokens[s, i, p, rep] = entry
    return ZooData(
        tuple(names), np.array(family_index, dtype=np.int64), np.array(slots, dtype=np.int64),
        np.array(truth, dtype=np.int64), tuple(chosen), tuple(prompt_ids), stops, ll_sum,
        ll_tokens, marked, marked_ext)


def paired_t(differences: FloatArray) -> float:
    """One-sample t of paired differences (0 with fewer than 2 points or no spread), as team D."""
    if len(differences) < 2:
        return 0.0
    spread = float(differences.std(ddof=1))
    return 0.0 if spread == 0.0 else float(differences.mean() / (spread / np.sqrt(len(differences))))


def per_token(total: FloatArray, tokens: FloatArray) -> FloatArray:
    """Per-cell mean log-likelihood per token (0 where a cell has no token), as team D."""
    safe = np.maximum(np.nan_to_num(tokens), 1.0)
    return np.where(np.nan_to_num(tokens) > 0, np.nan_to_num(total) / safe, 0.0)


def likelihood_flags(values: FloatArray, groups: Sequence[Sequence[int]], reference: str,
                     plain_position: Mapping[int, int] | None = None) -> BoolArray:
    """Disguise flags from per-token likelihoods `[E, M]` (cells shared by the endpoints).

    `reference == "plain"`: each endpoint against the plain endpoint of its group
    (`plain_position[group index]`); `"median"`: against the median of the other endpoints of its
    group. Flag iff |paired t| >= `LIKELIHOOD_T_LIMIT`.
    """
    flags = np.zeros(values.shape[0], dtype=np.bool_)
    for g, members in enumerate(groups):
        for i in members:
            if reference == "plain":
                assert plain_position is not None
                anchor = plain_position[g]
                if i == anchor:
                    continue
                ref = values[anchor]
            else:
                others = [j for j in members if j != i]
                if not others:
                    continue
                ref = np.median(values[others], axis=0)
            flags[i] = abs(paired_t(values[i] - ref)) >= LIKELIHOOD_T_LIMIT
    return flags


@dataclass(frozen=True)
class RelativeBand:
    """Team C's v4 band: flag iff rate >= max(floor, percentile of the group's rates + delta)."""

    delta: float = V4_DELTA
    floor: float = V4_FLOOR
    percentile: float = V4_BASELINE_PERCENTILE


def build_partition(
    rates: FloatArray, disguised: BoolArray, groups: Sequence[Sequence[int]], rule: Rule,
    relative: RelativeBand | None = None,
) -> tuple[Partition, BoolArray]:
    """Band rule per guessed family; a disguised endpoint is never isolated.

    The band is absolute (`rule`, v2) or, with `relative`, relative to each group (v4). Each
    flagged endpoint is its own model, every other endpoint of the group is the group's main
    model; model ids are unique across groups (families). Also returns the mask of isolated
    endpoints (flagged by the band and not disguised).
    """
    size = len(rates)
    family = np.zeros(size, dtype=np.int64)
    model = np.zeros(size, dtype=np.int64)
    isolated = np.zeros(size, dtype=np.bool_)
    next_model = 0
    for f_id, members in enumerate(groups):
        main = next_model
        next_model += 1
        if relative is None:
            low, high = rule.band_low, rule.band_high
        else:
            baseline = float(np.percentile(rates[list(members)], relative.percentile))
            low, high = max(relative.floor, baseline + relative.delta), float("inf")
        for i in members:
            family[i] = f_id
            flagged = (not disguised[i]) and low <= rates[i] <= high
            if flagged:
                model[i] = next_model
                next_model += 1
                isolated[i] = True
            else:
                model[i] = main
    return Partition(family, model), isolated


def truth_partition(data: ZooData, positions: Sequence[int]) -> Partition:
    """The true two-level partition of the selected endpoints."""
    return Partition(data.family_index[list(positions)].copy(),
                     data.truth_model[list(positions)].copy())


def scope_positions(data: ZooData, families: Sequence[str], use_w8a8: bool = False) -> list[int]:
    """Endpoint positions of the nine slots (slot 2 replaced by slot 10 when `use_w8a8`)."""
    wanted = {s for s in BASE_SLOTS if not (use_w8a8 and s == 2)} | ({10} if use_w8a8 else set())
    allowed = {data.families.index(f) for f in families if f in data.families}
    return [i for i in range(data.endpoint_count)
            if int(data.family_index[i]) in allowed and int(data.slot[i]) in wanted]


@dataclass
class CellSummary:
    """Score distribution of one (scope, rule, N) cell."""

    scope: str
    rule: str
    n: int
    scores: FloatArray
    isolated: FloatArray
    flagged: FloatArray
    family_correct: FloatArray
    positions: tuple[int, ...]

    def summary(self, data: ZooData) -> dict[str, object]:
        """JSON-friendly summary (median, 80 % interval, mean, share >= X and >= 0.6)."""
        low, high = np.percentile(self.scores, PERCENTILES)
        return {
            "scope": self.scope, "rule": self.rule, "n": self.n,
            "median": float(np.median(self.scores)), "mean": float(self.scores.mean()),
            "p10": float(low), "p90": float(high),
            "share_at_or_above_0.40": float(np.mean(self.scores >= THRESHOLD_X)),
            "share_at_or_above_0.60": float(np.mean(self.scores >= TEAM_DECISION_SCORE)),
            "isolated_share": {data.endpoints[p]: float(v)
                               for p, v in zip(self.positions, self.isolated, strict=True)},
            "disguise_flag_share": {data.endpoints[p]: float(v)
                                    for p, v in zip(self.positions, self.flagged, strict=True)},
            "family_correct_share": {data.endpoints[p]: float(v) for p, v in
                                     zip(self.positions, self.family_correct, strict=True)},
        }


def _guess_groups(rule_scope: str, data: ZooData, positions: Sequence[int], cells: tuple[IntArray, IntArray]
                  ) -> tuple[list[list[int]], IntArray]:
    """Groups (as scope-local positions) and the scorer index of every endpoint."""
    local_family = data.family_index[list(positions)]
    if rule_scope != SCOPE_LIKELIHOOD:
        scorers = local_family.copy()
    else:
        prompts_idx, repeats_idx = cells
        totals = data.ll_sum[:, list(positions)][:, :, prompts_idx, repeats_idx]
        tokens = data.ll_tokens[:, list(positions)][:, :, prompts_idx, repeats_idx]
        mean_ll = np.nansum(totals, axis=2) / np.maximum(np.nansum(tokens, axis=2), 1.0)
        mean_ll = np.where(np.isnan(mean_ll), -np.inf, mean_ll)
        scorers = np.argmax(mean_ll, axis=0).astype(np.int64)
    groups: dict[int, list[int]] = defaultdict(list)
    for local, scorer in enumerate(scorers.tolist()):
        groups[scorer].append(local)
    return list(groups.values()), scorers


def run_scope(
    data: ZooData, positions: Sequence[int], scope: str, n: int, replicates: int,
    rng: np.random.Generator, rule: Rule | None = None, identity_k: int = IDENTITY_K,
) -> list[CellSummary]:
    """Paired bootstrap of every rule on one scope: N arithmetic cells, k identity units.

    Each replicate draws `n` distinct arithmetic prompts and ONE repeat index per prompt shared by
    all endpoints (repeats are i.i.d. draws, so sharing the index loses nothing and pairs the
    likelihood cells), and `identity_k` identity units per endpoint without replacement. The
    `*40` likelihood rules use 40 cells drawn the same way from all (prompt, repeat) cells.
    """
    frozen = rule or Rule(BAND_LOW, threshold_x=THRESHOLD_X)
    local = list(positions)
    size = len(local)
    truth = truth_partition(data, local)
    stops = data.stops[local]
    marked = data.identity_marked[local]
    marked_ext = data.identity_marked_ext[local]
    unit_count = marked.shape[1]
    draws = unit_count // V4_IDENTITY_UNITS
    prompt_count, repeat_count = stops.shape[1], stops.shape[2]
    cell_count = prompt_count * repeat_count
    large = min(LIKELIHOOD_LARGE_CELLS, cell_count)
    true_main = np.array([variant_for_slot(int(s)).truth_model == MODEL_MAIN
                          for s in data.slot[local]])
    rows = np.arange(size)[:, None]
    scores = {r: np.empty(replicates) for r in RULES}
    isolated = {r: np.zeros(size) for r in RULES}
    flagged = {r: np.zeros(size) for r in RULES}
    family_correct = np.zeros(size)
    for replicate in range(replicates):
        prompts_idx = rng.choice(prompt_count, size=n, replace=False)
        repeats_idx = rng.integers(0, repeat_count, size=n)
        rates = stops[:, prompts_idx, repeats_idx].mean(axis=1)
        units = rng.random((size, unit_count)).argsort(axis=1)[:, :identity_k]
        disguised = marked[rows, units].mean(axis=1) > 0.5
        disguised_ext = marked_ext[rows, units].mean(axis=1) > 0.5
        stratified = (np.arange(V4_IDENTITY_UNITS) * draws
                      + rng.integers(0, draws, size=(size, V4_IDENTITY_UNITS)))
        disguised_v4 = marked[rows, stratified].mean(axis=1) > 0.5
        disguised_v4_ext = marked_ext[rows, stratified].mean(axis=1) > 0.5
        groups, scorers = _guess_groups(scope, data, local, (prompts_idx, repeats_idx))
        family_correct += scorers == data.family_index[local]
        flag_plain, flag_median = _likelihood_pair(
            data, local, groups, scorers, prompts_idx, repeats_idx)
        wide = rng.choice(cell_count, size=large, replace=False)
        _, flag_median40 = _likelihood_pair(
            data, local, groups, scorers, wide // repeat_count, wide % repeat_count)
        flags: dict[str, BoolArray] = {
            "v2": np.zeros(size, dtype=np.bool_), "v3": disguised, "v3ext": disguised_ext,
            "v4": disguised_v4, "v4ext": disguised_v4_ext,
            "lik_plain": flag_plain, "lik_median": flag_median, "lik_median40": flag_median40,
            "v3ext+lik_median": disguised_ext | flag_median,
            "v3ext+lik_median40": disguised_ext | flag_median40, "oracle": true_main,
        }
        for name in RULES:
            guess, isolated_mask = build_partition(
                rates, flags[name], groups, frozen,
                RelativeBand() if name.startswith("v4") else None)
            scores[name][replicate] = official_score(truth, guess)
            isolated[name] += isolated_mask
            flagged[name] += flags[name]
    return [CellSummary(scope, name, n, scores[name], isolated[name] / replicates,
                        flagged[name] / replicates, family_correct / replicates, tuple(local))
            for name in RULES]


def _likelihood_pair(
    data: ZooData, local: Sequence[int], groups: Sequence[Sequence[int]], scorers: IntArray,
    prompts_idx: IntArray, repeats_idx: IntArray,
) -> tuple[BoolArray, BoolArray]:
    """`(plain-reference flags, median-reference flags)` on the chosen cells."""
    values = _likelihood_matrix(data, local, scorers, prompts_idx, repeats_idx)
    plain_position: dict[int, int] = {}
    for g, members in enumerate(groups):
        anchors = [i for i in members if int(data.slot[local[i]]) == 1]
        if anchors:
            plain_position[g] = anchors[0]
    plain = (likelihood_flags(values, groups, "plain", plain_position)
             if len(plain_position) == len(groups) else np.zeros(len(local), dtype=np.bool_))
    return plain, likelihood_flags(values, groups, "median")


def _likelihood_matrix(
    data: ZooData, local: Sequence[int], scorers: IntArray, prompts_idx: IntArray,
    repeats_idx: IntArray,
) -> FloatArray:
    """Per-token log-likelihood `[E, M]` of the chosen cells under each endpoint's scorer."""
    values = np.zeros((len(local), len(prompts_idx)))
    for i, position in enumerate(local):
        s = int(scorers[i])
        total = data.ll_sum[s, position][prompts_idx, repeats_idx]
        tokens = data.ll_tokens[s, position][prompts_idx, repeats_idx]
        values[i] = per_token(total, tokens)
    return values


def likelihood_tables(directory: Path, data: ZooData, overlay: Path | None = None
                      ) -> dict[str, object]:
    """Team D's likelihood statistics on ALL arithmetic cells of the zoo (no sampling).

    Per family and endpoint, under the family's own base model: the mean per-token log-likelihood,
    its paired t against the plain endpoint (cells = prompt x repeat; |t| >= 3 is the disguise
    limit), and for each fp8 simulation the mean per-token likelihood ratio
    `Lambda = ll_fp8 - ll_base` with its paired t against the plain endpoint (fp8 is "seen" when
    t >= 3). Nothing is fitted.
    """
    tables: dict[str, object] = {}
    for family in data.families:
        payload = read_likelihood_file(directory, family)
        if payload is None:
            continue
        scores = {mode: dict(table) for mode, table in payload["scores"].items()}
        extra = read_likelihood_file(overlay, family) if overlay is not None else None
        if extra is not None:
            for mode, table in extra["scores"].items():
                scores.setdefault(mode, {}).update(table)
        rows: dict[str, dict[str, FloatArray]] = {}
        for name in (n for n in data.endpoints if n.startswith(f"zoo-{family}-")):
            shape = (len(data.prompts), EXPECTED_REPEATS)
            arrays = {k: np.full(shape, np.nan) for k in ("base", "tok", *FP8_MODES)}
            for p, pid in enumerate(data.prompts):
                for rep in range(EXPECTED_REPEATS):
                    key = f"{name}|{pid}|{rep}"
                    if key not in scores.get("base", {}):
                        continue
                    arrays["base"][p, rep], arrays["tok"][p, rep] = scores["base"][key]
                    for mode in FP8_MODES:
                        if key in scores.get(mode, {}):
                            arrays[mode][p, rep] = scores[mode][key][0]
            rows[name] = arrays
        plain = f"zoo-{family}-01"
        if plain not in rows:
            continue
        per_endpoint: dict[str, object] = {}
        flat_plain = _per_cell(rows[plain])
        for name, arrays in rows.items():
            mine = _per_cell(arrays)
            entry: dict[str, object] = {
                "variant": variant_for_slot(int(name.rsplit("-", 1)[1])).key,
                "mean_ll_per_token": _mean_ratio(arrays["base"], arrays["tok"]),
                "t_base_ll_vs_plain": paired_t(mine["base"] - flat_plain["base"]),
            }
            for mode in FP8_MODES:
                entry[f"mean_lambda_{mode}"] = _mean_ratio(arrays[mode] - arrays["base"],
                                                           arrays["tok"])
                entry[f"t_lambda_{mode}_vs_plain"] = paired_t(mine[mode] - flat_plain[mode])
            per_endpoint[name] = entry
        tables[family] = per_endpoint
    return tables


def _per_cell(arrays: Mapping[str, FloatArray]) -> dict[str, FloatArray]:
    """Per-cell per-token base likelihood and fp8 ratios, flattened (cells without token = 0)."""
    tokens = np.nan_to_num(arrays["tok"]).ravel()
    safe = np.maximum(tokens, 1.0)
    base = np.nan_to_num(arrays["base"]).ravel()
    out = {"base": np.where(tokens > 0, base / safe, 0.0)}
    for mode in FP8_MODES:
        ratio = np.nan_to_num(arrays[mode]).ravel() - base
        out[mode] = np.where(tokens > 0, ratio / safe, 0.0)
    return out


def _mean_ratio(total: FloatArray, tokens: FloatArray) -> float | None:
    """Token-weighted mean of `total / tokens`, `None` without any token."""
    count = float(np.nansum(tokens))
    return float(np.nansum(total) / count) if count > 0 else None


def variant_tables(data: ZooData) -> dict[str, object]:
    """Per (family, variant) descriptive statistics on ALL answers (no sampling).

    `full_stop_rate`: share of arithmetic answers ending with a full stop; `identity_marked`:
    share of identity answers naming no base model (frozen lexicon / with Mistral);
    `mean_ll_per_token`: mean per-token log-likelihood under the family's own base model.
    """
    table: dict[str, object] = {}
    for i, name in enumerate(data.endpoints):
        scorer = int(data.family_index[i])
        total = np.nansum(data.ll_sum[scorer, i])
        tokens = np.nansum(data.ll_tokens[scorer, i])
        table[name] = {
            "slot": int(data.slot[i]), "variant": variant_for_slot(int(data.slot[i])).key,
            "full_stop_rate": float(data.stops[i].mean()),
            "identity_marked": float(data.identity_marked[i].mean()),
            "identity_marked_ext": float(data.identity_marked_ext[i].mean()),
            "mean_ll_per_token": float(total / tokens) if tokens > 0 else None,
        }
    return table


def ceilings(data: ZooData, positions: Sequence[int]) -> dict[str, float]:
    """Reference scores of the scope: nothing found, fp8 lumped with S, everything found."""
    local = list(positions)
    truth = truth_partition(data, local)
    lumped_model = truth.model.copy()
    for i, p in enumerate(local):
        if variant_for_slot(int(data.slot[p])).key.startswith("fp8"):
            main = next(truth.model[j] for j, q in enumerate(local)
                        if data.family_index[q] == data.family_index[p]
                        and variant_for_slot(int(data.slot[q])).truth_model == MODEL_MAIN)
            lumped_model[i] = main
    everything_one = Partition(truth.family.copy(), truth.family.copy())
    return {
        "one_model_per_family": official_score(truth, everything_one),
        "fp8_lumped_with_S": official_score(truth, Partition(truth.family, lumped_model)),
        "everything_found": official_score(truth, truth),
    }


def evaluate_all(
    data: ZooData, replicates: int = REPLICATES, sizes: Sequence[int] = SAMPLE_SIZES,
    seed: int = SEED,
) -> dict[str, object]:
    """Every scope x N x rule; the payload written to `evaluation.json`."""
    rng = np.random.default_rng(seed)
    cells: list[dict[str, object]] = []
    scopes: list[tuple[str, list[str], bool]] = [(SCOPE_SINGLE, [f], False) for f in data.families]
    if len(data.families) > 1:
        scopes += [(SCOPE_ORACLE, list(data.families), False),
                   (SCOPE_LIKELIHOOD, list(data.families), False)]
    scopes += [(SCOPE_ORACLE + "_w8a8", list(data.families), True)]
    scopes += [(SCOPE_SINGLE, [f], True) for f in data.families]
    reference: dict[str, object] = {}
    for scope, families, w8a8 in scopes:
        positions = scope_positions(data, families, w8a8)
        base_scope = scope.removesuffix("_w8a8")
        label = scope if len(families) > 1 else f"{SCOPE_SINGLE}:{families[0]}"
        if w8a8 and len(families) == 1:
            label += W8A8_SUFFIX
        reference[label] = ceilings(data, positions)
        for n in sizes:
            for cell in run_scope(data, positions, base_scope, n, replicates, rng):
                summary = cell.summary(data)
                summary["scope"] = label
                cells.append(summary)
    return {"seed": seed, "replicates": replicates, "band_low": BAND_LOW,
            "likelihood_cells": int(data.stops.shape[1] * data.stops.shape[2]),
            "threshold_x": THRESHOLD_X, "decision_score": TEAM_DECISION_SCORE,
            "identity_k": IDENTITY_K, "likelihood_t_limit": LIKELIHOOD_T_LIMIT,
            "families": list(data.families), "ceilings": reference, "cells": cells,
            "variants": variant_tables(data)}


def variant_key_of(endpoint: str) -> str:
    """The variant key of a zoo endpoint name."""
    slot = int(endpoint.rsplit("-", 1)[1])
    return variant_for_slot(slot).key


assert all(key in VARIANT_BY_KEY for key in ("plain", "fp8_w", "fp8_w8a8"))
