"""Driver of team E's Mac-side analysis: matrix, identification and temperature profiles.

    python -m hackathon.model_equality.research.e_run matrix \
        --ll-dir <data/ll copied from JURECA> --answers-dir artifacts/model_equality --out-dir <dir>

Reads the `.npz` containers of the GPU task, writes CSV/JSON tables and prints a short summary.
Pure post-processing (numpy): a few seconds, no model is loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from hackathon.model_equality.research.e_analysis import (
    AnalysisError,
    llr_summary,
    paired_per_token_comparison,
    temperature_profile,
    tokens_needed,
)
from hackathon.model_equality.research.e_calib import (
    CLASS_OF,
    RowFeatureTable,
    build_row_table,
    draw_features,
    separation_table,
    write_table,
)
from hackathon.model_equality.research.e_classify import ClassificationResult, classify
from hackathon.model_equality.research.e_loglik import (
    EP_ENDPOINT_PREFIX,
    FP8_VARIANTS,
    STAGE_FP8,
    STAGE_MATRIX,
    TEMPLATE_DEFAULT,
    TEMPLATE_NOSYS,
    VARIANT_BASE,
    RowScores,
    load_answers,
    load_scores,
)
from hackathon.model_equality.research.e_matrix import (
    Cell,
    Identification,
    build_cells,
    identify,
    load_containers,
    row_sums,
    write_matrix_csv,
)
from hackathon.model_equality.research.e_models import CANDIDATE_BY_KEY
from hackathon.model_equality.research.e_partition import (
    FP8_LLR_VARIANT,
    EndpointEvidence,
    all_identity_answers,
    full_stop_shares,
    identity_answers,
    propose,
    temperature_one_threshold,
    write_partition,
)
from hackathon.model_equality.research.e_sim import load_sim
from hackathon.model_equality.research.e_zoo import (
    CLEAN_CLASSES as ZOO_CLEAN_CLASSES,
)
from hackathon.model_equality.research.e_zoo import (
    REAL_FP8_EFFECT_SCALE,
    SIM_BUDGETS,
    attainable_budgets,
    load_endpoint_classes,
    simulate_families,
)
from hackathon.model_equality.research.e_zoo import (
    TARGET_CLASSES as ZOO_TARGET_CLASSES,
)
from hackathon.model_equality.research.e_zoo_ll import ll_separation_rows, load_ll_rows
from hackathon.model_equality.store import Answer

IDENT_FIELDS: tuple[str, ...] = (
    "endpoint", "group", "best", "family", "template", "runner_up", "margin_nats", "win_share",
    "bootstrap_win", "confidence", "best_other_family", "gap_other_family_nats",
    "best_family_per_char", "second_family", "second_family_gap_nats",
)
PROFILE_FIELDS: tuple[str, ...] = (
    "endpoint", "group", "candidate", "template", "n_tokens", "z_at_1", "mle_temperature",
    "mle_at_boundary", "zero_crossing_temperature", "ll_gain_nats", "z_curve", "ll_curve",
)
GROUP_EP = "ep"
GROUP_REF = "ref"
PLAIN_REFERENCE_ENDPOINT = "ref-01"
SIM_FIELDS: tuple[str, ...] = (
    "candidate", "variant", "rows", "tokens_h0", "tokens_h1", "mu0", "mu1", "sd0", "sd1", "gap",
    "tokens_z1.645", "tokens_z2.326", "tokens_z3", "long_answers_z2.326",
)
LONG_ANSWER_TOKENS = 254
"""Mean length of the two long ep answers (long-explain-000, long-story-000), capped at 256."""
SIM_Z_LEVELS: tuple[float, ...] = (1.645, 2.326, 3.0)
FP8_FIELDS: tuple[str, ...] = (
    "candidate", "variant", "endpoint", "group", "n_rows", "n_tokens", "llr_mean_per_token",
    "llr_se", "llr_t", "llr_sd_per_token", "paired_t_vs_ref01", "paired_mean_vs_ref01",
    "tokens_for_1pct",
)


def group_of(endpoint: str) -> str:
    """`ep` or `ref` from the endpoint id."""
    return GROUP_EP if endpoint.startswith(EP_ENDPOINT_PREFIX) else GROUP_REF


def family_gaps(
    cells: Mapping[str, Cell], families: Mapping[str, str]
) -> list[tuple[str, str, float]]:
    """Best candidate of each family for one endpoint: `[(family, candidate, total_ll)]`, best first."""
    best_per_family: dict[str, tuple[str, float]] = {}
    for candidate, cell in cells.items():
        family = families[candidate]
        if family not in best_per_family or cell.total_ll > best_per_family[family][1]:
            best_per_family[family] = (candidate, cell.total_ll)
    ranked = sorted(best_per_family.items(), key=lambda item: -item[1][1])
    return [(family, candidate, total) for family, (candidate, total) in ranked]


def identification_rows(
    cells: Mapping[str, Mapping[str, Cell]], families: Mapping[str, str]
) -> list[dict[str, Any]]:
    """One table row per endpoint with the winner, its margin and the family-level margin."""
    rows: list[dict[str, Any]] = []
    for endpoint, per_candidate in cells.items():
        if len(per_candidate) < 2:
            continue
        ident: Identification = identify(endpoint, per_candidate, families)
        ranking = family_gaps(per_candidate, families)
        second_family = ranking[1][0] if len(ranking) > 1 else ""
        second_gap = ranking[0][2] - ranking[1][2] if len(ranking) > 1 else float("nan")
        rows.append({
            "endpoint": endpoint, "group": group_of(endpoint), "best": ident.best,
            "family": families[ident.best], "template": per_candidate[ident.best].template,
            "runner_up": ident.runner_up, "margin_nats": round(ident.margin_nats, 2),
            "win_share": round(ident.win_share, 3), "bootstrap_win": round(ident.bootstrap_win, 3),
            "confidence": ident.confidence, "best_other_family": ident.best_other_family[0],
            "gap_other_family_nats": round(ident.best_other_family[1], 2),
            "best_family_per_char": round(per_candidate[ident.best].per_char, 4),
            "second_family": second_family, "second_family_gap_nats": round(second_gap, 2),
        })
    return rows


def write_rows(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a CSV with a fixed column order."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def profile_rows(
    identification: Sequence[Mapping[str, Any]],
    containers: Mapping[str, tuple[Mapping[str, object], Mapping[tuple[str, str], RowScores]]],
) -> list[dict[str, Any]]:
    """Temperature profile of every endpoint under its identified candidate and template."""
    rows: list[dict[str, Any]] = []
    for ident in identification:
        header, per_variant = containers[ident["best"]]
        scores = per_variant[(ident["template"], "base")]
        endpoints = header["endpoint"]
        assert isinstance(endpoints, list)
        selected = [i for i, endpoint in enumerate(endpoints) if endpoint == ident["endpoint"]]
        try:
            profile = temperature_profile(scores, selected)
        except AnalysisError:
            continue
        unit = profile.temperatures.index(1.0)
        rows.append({
            "endpoint": ident["endpoint"], "group": ident["group"], "candidate": ident["best"],
            "template": ident["template"], "n_tokens": profile.n_tokens,
            "z_at_1": round(float(profile.z[unit]), 2),
            "mle_temperature": round(profile.mle_temperature, 3),
            "mle_at_boundary": profile.mle_at_boundary,
            "zero_crossing_temperature": (
                None if profile.zero_crossing_temperature is None
                else round(profile.zero_crossing_temperature, 3)
            ),
            "ll_gain_nats": round(profile.ll_gain_nats, 2),
            "z_curve": json.dumps([[t, round(float(z), 2)] for t, z in zip(profile.temperatures, profile.z, strict=True)]),
            "ll_curve": json.dumps([[t, round(float(v), 1)] for t, v in zip(profile.temperatures, profile.total_logp, strict=True)]),
        })
    return rows


def matrix_command(args: argparse.Namespace) -> int:
    """Build and write the matrix, the identification and the temperature profiles."""
    answers: list[Answer] = load_answers(Path(args.answers_dir))
    containers = load_containers(Path(args.ll_dir), STAGE_MATRIX)
    if len(containers) < 2:
        print(f"need at least two candidates in {args.ll_dir}, found {sorted(containers)}")
        return 2
    endpoints = sorted({answer.endpoint for answer in answers})
    cells = build_cells(containers, answers, endpoints)
    families = {key: CANDIDATE_BY_KEY[key].family for key in containers}
    candidates = sorted(containers, key=lambda key: (CANDIDATE_BY_KEY[key].family,
                                                     CANDIDATE_BY_KEY[key].params_billion))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_matrix_csv(out_dir / "matrix_all.csv", cells, candidates)
    rows = identification_rows(cells, families)
    write_rows(out_dir / "identification.csv", IDENT_FIELDS, rows)
    profiles = profile_rows(rows, containers)
    if profiles:
        write_rows(out_dir / "temperature_profiles.csv", PROFILE_FIELDS, profiles)
    (out_dir / "identification.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    for group in (GROUP_REF, GROUP_EP):
        counts: dict[str, int] = {}
        for row in rows:
            if row["group"] == group:
                counts[row["best"]] = counts.get(row["best"], 0) + 1
        print(group, "winners:", dict(sorted(counts.items(), key=lambda item: -item[1])))
    low = [row["endpoint"] for row in rows if row["confidence"] == "low"]
    print("low-confidence endpoints:", low)
    print(f"candidates: {len(candidates)}, endpoints: {len(rows)}; tables in {out_dir}")
    return 0


def fp8_command(args: argparse.Namespace) -> int:
    """Per endpoint log-likelihood ratio of every simulated fp8 variant against the base."""
    containers = load_containers(Path(args.ll_dir), args.stage)
    identified: dict[str, str] = {}
    ident_path = Path(args.identification)
    if ident_path.is_file():
        for row in json.loads(ident_path.read_text(encoding="utf-8")):
            identified[row["endpoint"]] = row["best"]
    rows: list[dict[str, Any]] = []
    for candidate, (header, per_variant) in containers.items():
        base = per_variant.get((TEMPLATE_DEFAULT, VARIANT_BASE))
        if base is None:
            continue
        endpoints = header["endpoint"]
        ids = header["prompt_id"]
        repeats = header["repeat"]
        assert isinstance(endpoints, list) and isinstance(ids, list) and isinstance(repeats, list)
        base_sum = row_sums(base, "logp")
        tokens = base.n_tokens.astype(np.int64)
        cells = [(str(i), int(r)) for i, r in zip(ids, repeats, strict=True)]
        for variant in FP8_VARIANTS:
            scores = per_variant.get((TEMPLATE_DEFAULT, variant))
            if scores is None:
                continue
            delta = row_sums(scores, "logp") - base_sum
            by_endpoint = {
                endpoint: [i for i, name in enumerate(endpoints) if name == endpoint]
                for endpoint in sorted(set(endpoints))
            }
            reference = by_endpoint.get(PLAIN_REFERENCE_ENDPOINT)
            for endpoint, index in by_endpoint.items():
                if endpoint.startswith(EP_ENDPOINT_PREFIX) and identified.get(endpoint) != candidate:
                    continue
                summary = llr_summary(delta[index], tokens[index])
                paired_t = paired_mean = None
                if reference is not None and endpoint != PLAIN_REFERENCE_ENDPOINT:
                    comparison = paired_per_token_comparison(
                        [cells[i] for i in index], delta[index], tokens[index],
                        [cells[i] for i in reference], delta[reference], tokens[reference],
                    )
                    paired_t, paired_mean = comparison.t_statistic, comparison.mean_difference
                needed = tokens_needed(abs(summary.mean_per_token), summary.sd_per_token)
                rows.append({
                    "candidate": candidate, "variant": variant, "endpoint": endpoint,
                    "group": group_of(endpoint), "n_rows": len(index), "n_tokens": summary.n_tokens,
                    "llr_mean_per_token": round(summary.mean_per_token, 5),
                    "llr_se": round(summary.standard_error, 5), "llr_t": round(summary.t_statistic, 2),
                    "llr_sd_per_token": round(summary.sd_per_token, 4),
                    "paired_t_vs_ref01": None if paired_t is None else round(paired_t, 2),
                    "paired_mean_vs_ref01": None if paired_mean is None else round(paired_mean, 5),
                    "tokens_for_1pct": None if needed is None else round(needed),
                })
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "fp8_llr.csv", FP8_FIELDS, rows)
    print(f"{len(rows)} rows in {out_dir / 'fp8_llr.csv'}")
    return 0


def llr_by_variant(
    header: Mapping[str, object], per_variant: Mapping[tuple[str, str], RowScores], prefix: str = ""
) -> tuple[list[tuple[str, str, int]], dict[str, Any]]:
    """`(row keys, {prefix + variant: per-row LLR sums})` of an fp8-stage container (default template)."""
    endpoints, ids, repeats = header["endpoint"], header["prompt_id"], header["repeat"]
    assert isinstance(endpoints, list) and isinstance(ids, list) and isinstance(repeats, list)
    keys = [(str(a), str(b), int(c)) for a, b, c in zip(endpoints, ids, repeats, strict=True)]
    base = row_sums(per_variant[(TEMPLATE_DEFAULT, VARIANT_BASE)], "logp")
    out: dict[str, Any] = {}
    for (template, variant), scores in per_variant.items():
        if template == TEMPLATE_DEFAULT and variant != VARIANT_BASE:
            out[prefix + variant] = row_sums(scores, "logp") - base
    return keys, out


def build_calibration_table(
    ll_dir: Path, candidate: str, templates: Mapping[str, str], stages: Sequence[str],
) -> RowFeatureTable:
    """Row feature table of one candidate: tempered sums per endpoint template, fp8 LLR per row.

    `templates` maps an endpoint to the matrix template that explains it best.
    """
    header, per_variant = load_scores(ll_dir / f"ll_{STAGE_MATRIX}_{candidate}.npz")
    endpoints = header["endpoint"]
    keys: list[tuple[str, str, int]] = [
        (str(e), str(i), int(r)) for e, i, r in zip(endpoints, header["prompt_id"], header["repeat"],
                                                   strict=True)
    ]
    index_of = {key: position for position, key in enumerate(keys)}
    llr: dict[str, Any] = {}
    for stage in stages:
        path = ll_dir / f"ll_{stage}_{candidate}.npz"
        if not path.is_file():
            continue
        stage_header, stage_variants = load_scores(path)
        stage_keys, stage_llr = llr_by_variant(stage_header, stage_variants, prefix=f"{stage}_")
        for name, values in stage_llr.items():
            aligned = np.full(len(keys), np.nan)
            for position, key in enumerate(stage_keys):
                if key in index_of:
                    aligned[index_of[key]] = values[position]
            llr[name] = aligned
    tables = {
        template: build_row_table(keys, per_variant[(template, VARIANT_BASE)], llr)
        for template in (TEMPLATE_DEFAULT, TEMPLATE_NOSYS)
    }
    chosen = np.array([templates.get(key[0], TEMPLATE_DEFAULT) == TEMPLATE_NOSYS for key in keys])
    default, nosys = tables[TEMPLATE_DEFAULT], tables[TEMPLATE_NOSYS]
    pick = chosen[:, None]
    return RowFeatureTable(
        keys=default.keys, n_tokens=default.n_tokens, temperatures=default.temperatures,
        lp=np.where(pick, nosys.lp, default.lp), ent=np.where(pick, nosys.ent, default.ent),
        var=np.where(pick, nosys.var, default.var), llr=default.llr,
    )


def calib_command(args: argparse.Namespace) -> int:
    """Separation of fp8, LoRA and disguises from clean endpoints at several token budgets (ref)."""
    identification = json.loads(Path(args.identification).read_text(encoding="utf-8"))
    templates = {row["endpoint"]: row["template"] for row in identification}
    table = build_calibration_table(Path(args.ll_dir), args.candidate, templates, args.fp8_stages)
    rows = separation_table(table, draws=args.draws)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_table(out_dir / "calibration_auc.csv", rows)
    print(f"{len(rows)} rows in {out_dir / 'calibration_auc.csv'}; features: "
          f"{sorted({str(r['feature']) for r in rows})}")
    return 0


def classify_command(args: argparse.Namespace) -> int:
    """Row-disjoint supervised read-out of the features on the ref endpoints (model labels)."""
    identification = json.loads(Path(args.identification).read_text(encoding="utf-8"))
    templates = {row["endpoint"]: row["template"] for row in identification}
    table = build_calibration_table(Path(args.ll_dir), args.candidate, templates, args.fp8_stages)
    llr_names = sorted(table.llr)
    feature_sets: dict[str, list[str]] = {
        "temperature+calibration": ["mle_temperature", "z_at_one", "mean_ll"],
    }
    for name in llr_names:
        feature_sets[f"+{name}"] = ["mle_temperature", "z_at_one", "mean_ll", f"llr_{name}"]
    if llr_names:
        feature_sets["+all llr"] = ["mle_temperature", "z_at_one", "mean_ll",
                                    *[f"llr_{name}" for name in llr_names]]
    results: list[dict[str, Any]] = []
    for tokens in args.budgets:
        for label, features in feature_sets.items():
            outcome: ClassificationResult = classify(table, tokens, features)
            results.append({
                "tokens": tokens, "feature_set": label, "balanced_accuracy": round(outcome.balanced_accuracy, 3),
                **{f"recall_{key}": round(value, 3) for key, value in outcome.recall.items()},
                "confusion": outcome.confusion.tolist(),
            })
            print(tokens, label, "balanced", round(outcome.balanced_accuracy, 3),
                  {k: round(v, 2) for k, v in outcome.recall.items()}, flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "classification_ref.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


def partition_command(args: argparse.Namespace) -> int:
    """Proposed families and model classes of the ep endpoints from the tables already written."""
    data_dir = Path(args.data_dir)
    identification = json.loads((data_dir / "identification.json").read_text(encoding="utf-8"))
    with (data_dir / "temperature_profiles.csv").open(encoding="utf-8") as handle:
        profiles = {row["endpoint"]: row for row in csv.DictReader(handle)}
    fp8_llr: dict[str, float] = {}
    with (data_dir / "fp8_llr.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["group"] == GROUP_EP and row["variant"] == FP8_LLR_VARIANT:
                fp8_llr[row["endpoint"]] = float(row["llr_mean_per_token"])
    answers = load_answers(Path(args.answers_dir), rows="ep")
    identity, stops = identity_answers(answers), full_stop_shares(answers)
    identity_lists = all_identity_answers(answers)
    evidence = [
        EndpointEvidence(
            endpoint=row["endpoint"], family=row["best"], family_confidence=row["confidence"],
            mle_temperature=float(profiles[row["endpoint"]]["mle_temperature"]),
            z_at_one=float(profiles[row["endpoint"]]["z_at_1"]),
            fp8_llr=fp8_llr.get(row["endpoint"]), identity_answer=identity.get(row["endpoint"], ""),
            full_stop_share=stops.get(row["endpoint"]),
            n_tokens=int(profiles[row["endpoint"]]["n_tokens"]),
            identity_answers=identity_lists.get(row["endpoint"], ()),
        )
        for row in identification if row["group"] == GROUP_EP
    ]
    rows = propose(evidence)
    write_partition(data_dir / args.out_name, rows)
    by_model: dict[str, list[str]] = {}
    for row in rows:
        by_model.setdefault(str(row["model_id"]), []).append(str(row["endpoint"]))
    for model_id, members in sorted(by_model.items()):
        print(model_id, members)
    return 0


def sim_command(args: argparse.Namespace) -> int:
    """Effect size and token need of every fp8 variant on the simulated endpoints (per base)."""
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(args.ll_dir).glob("sim_*.npz")):
        meta, sums, tokens = load_sim(path)
        candidate = str(meta["candidate"])
        variants = [v for v in meta["variants"] if (v, v) in sums]
        for variant in variants:
            h0 = sums[(VARIANT_BASE, variant)] - sums[(VARIANT_BASE, VARIANT_BASE)]
            h1 = sums[(variant, variant)] - sums[(variant, VARIANT_BASE)]
            n0, n1 = tokens[VARIANT_BASE], tokens[variant]
            s0, s1 = llr_summary(h0, n0), llr_summary(h1, n1)
            gap = s1.mean_per_token - s0.mean_per_token
            row: dict[str, Any] = {
                "candidate": candidate, "variant": variant, "rows": len(h0),
                "tokens_h0": s0.n_tokens, "tokens_h1": s1.n_tokens,
                "mu0": round(s0.mean_per_token, 5), "mu1": round(s1.mean_per_token, 5),
                "sd0": round(s0.sd_per_token, 4), "sd1": round(s1.sd_per_token, 4),
                "gap": round(gap, 5),
            }
            for z_level in SIM_Z_LEVELS:
                needed = None if gap <= 0 else (z_level * (s0.sd_per_token + s1.sd_per_token) / gap) ** 2
                row[f"tokens_z{z_level:g}"] = None if needed is None else round(needed)
            two = row["tokens_z2.326"]
            row["long_answers_z2.326"] = None if two is None else round(two / LONG_ANSWER_TOKENS, 1)
            rows.append(row)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "sim_power.csv", SIM_FIELDS, rows)
    for row in rows:
        print({k: row[k] for k in ("candidate", "variant", "mu0", "mu1", "sd0", "gap", "tokens_z1.645",
                                   "tokens_z2.326", "long_answers_z2.326")})
    return 0


REALISM_FIELDS: tuple[str, ...] = (
    "variant", "mu_clean", "mu_ref02", "real_gap", "sim_gap", "real_over_sim", "tokens_z2.326_sim",
    "tokens_z2.326_real_scale",
)
SIM_VARIANT_OF_STAGE_VARIANT: Mapping[str, str] = {
    "fp8_w_tensor": "w_tensor", "fp8_w_channel": "w_channel", "fp8_w8a8_token": "w8a8_token",
    "fp8x_w_block128": "w_block128",
}


def realism_command(args: argparse.Namespace) -> int:
    """How close the real fp8 endpoint (ref-02) is to each simulated variant, and what it costs.

    For every simulated variant V: `real_gap` is the mean per-token log-likelihood ratio of ref-02
    minus that of the clean ref endpoints (ref-01, ref-06); `sim_gap` is the gap between the
    simulated fp8 endpoint (H1) and the simulated plain endpoint (H0) of `sim_power.csv`. A ratio
    below 1 means the real fp8 is milder than the simulation, so the token needs of the simulation
    are optimistic by the square of the inverse ratio.
    """
    identification = json.loads(Path(args.identification).read_text(encoding="utf-8"))
    templates = {row["endpoint"]: row["template"] for row in identification}
    table = build_calibration_table(Path(args.ll_dir), args.candidate, templates, args.fp8_stages)
    sim_rows = {
        (row["candidate"], row["variant"]): row
        for row in csv.DictReader((Path(args.out_dir) / "sim_power.csv").open(encoding="utf-8"))
    }
    keys = table.keys

    def mean_llr(variant: str, endpoints: Sequence[str]) -> float:
        index = np.array([i for i, key in enumerate(keys) if key[0] in endpoints])
        values = table.llr[variant][index]
        tokens = table.n_tokens[index]
        valid = ~np.isnan(values)
        return float(values[valid].sum() / tokens[valid].sum())

    rows: list[dict[str, Any]] = []
    for variant in sorted(table.llr):
        clean = mean_llr(variant, ("ref-01", "ref-06"))
        fp8 = mean_llr(variant, ("ref-02",))
        real_gap = fp8 - clean
        sim_name = SIM_VARIANT_OF_STAGE_VARIANT.get(variant)
        sim = sim_rows.get((args.candidate, sim_name)) if sim_name else None
        sim_gap = None if sim is None else float(sim["gap"])
        tokens_sim = None if sim is None or sim["tokens_z2.326"] in ("", "None") else float(sim["tokens_z2.326"])
        ratio = None if sim_gap in (None, 0.0) or sim_gap is None else real_gap / sim_gap
        rows.append({
            "variant": variant, "mu_clean": round(clean, 5), "mu_ref02": round(fp8, 5),
            "real_gap": round(real_gap, 5), "sim_gap": None if sim_gap is None else round(sim_gap, 5),
            "real_over_sim": None if ratio is None else round(ratio, 3),
            "tokens_z2.326_sim": None if tokens_sim is None else round(tokens_sim),
            "tokens_z2.326_real_scale": (
                None if tokens_sim is None or not ratio or ratio <= 0 else round(tokens_sim / ratio**2)
            ),
        })
        print(rows[-1])
    write_rows(Path(args.out_dir) / "fp8_realism.csv", REALISM_FIELDS, rows)
    return 0


ZOO_FAMILIES: tuple[str, ...] = ("qwen15", "qwen3", "llama1", "smollm")
ZOO_LL_FIELDS: tuple[str, ...] = (
    "family", "fp8_effect_scale", "target", "feature", "tokens", "auc", "tpr_at_5pct_fpr",
    "mean_target", "mean_clean", "sd_clean",
)


def zoo_ll_command(args: argparse.Namespace) -> int:
    """fp8 and mean-likelihood separation on the zoo, from team F's likelihood files (no GPU)."""
    zoo_dir = Path(args.zoo_dir)
    rows_out: list[dict[str, Any]] = []
    for family in ZOO_FAMILIES:
        class_of = load_endpoint_classes(zoo_dir / f"meta_{family}.json")
        rows = load_ll_rows(zoo_dir / f"ll_{family}.json.gz", family)
        smallest = min(int(r.n_tokens.sum()) for r in rows.values())
        budgets = [b for b in SIM_BUDGETS if b <= 0.8 * smallest]
        print(family, "tokens per endpoint (min):", smallest, "budgets:", budgets, flush=True)
        for scale in (1.0, REAL_FP8_EFFECT_SCALE):
            for row in ll_separation_rows(rows, class_of, budgets, fp8_effect_scale=scale):
                rows_out.append({"family": family, "fp8_effect_scale": scale, **row})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "zoo_ll_separation.csv", ZOO_LL_FIELDS, rows_out)
    for row in rows_out:
        if row["feature"] == "llr_fp8_w" and row["target"] == "fp8":
            print({k: row[k] for k in ("family", "fp8_effect_scale", "tokens", "auc", "tpr_at_5pct_fpr")})
    return 0


PARTITION_SIM_FIELDS: tuple[str, ...] = (
    "source", "budget", "oracle_identity", "fp8_margin_sigma", "fp8_effect_scale", "mean_score", "sd_score",
    "median_score", "p10_score", "p90_score", "isolated_share",
)


def partition_sim_command(args: argparse.Namespace) -> int:
    """Official score (0.9 x MCC_model) of the partition rules on simulated families.

    Source `ref`: pseudo-endpoints drawn from the REAL ref endpoints (SmolLM2-360M, real fp8 and
    LoRA); the family has the ref design plus a second plain copy. Budgets are tokens per endpoint.
    """
    identification = json.loads(Path(args.identification).read_text(encoding="utf-8"))
    templates = {row["endpoint"]: row["template"] for row in identification}
    table = build_calibration_table(Path(args.ll_dir), args.candidate, templates, args.fp8_stages)
    fp8_feature = args.fp8_feature
    rows: list[dict[str, Any]] = []
    for budget in args.budgets:
        for oracle in (True, False):
            for margin in args.margins:
                result = simulate_families(
                    table, CLASS_OF, budget, oracle_identity=oracle, fp8_feature=fp8_feature,
                    fp8_margin_sigma=margin, trials=args.trials,
                )
                rows.append({
                    "source": "ref", "budget": budget, "oracle_identity": oracle,
                    "fp8_margin_sigma": margin, "fp8_effect_scale": 1.0,
                    "mean_score": round(result.mean_score, 3), "sd_score": round(result.sd_score, 3),
                    "median_score": round(result.median_score, 3), "p10_score": round(result.p10_score, 3),
                    "p90_score": round(result.p90_score, 3),
                    "isolated_share": json.dumps({k: round(v, 2) for k, v in result.isolated_share.items()}),
                })
                print(budget, "identity" if oracle else "no identity", "margin", margin,
                      "median", round(result.median_score, 3),
                      "p10-p90", (round(result.p10_score, 3), round(result.p90_score, 3)), flush=True)
    write_rows(Path(args.out_dir) / "partition_sim_ref.csv", PARTITION_SIM_FIELDS, rows)
    return 0


ZOO_FULL_SEP_FIELDS: tuple[str, ...] = ("family", *ZOO_LL_FIELDS[1:])


def best_templates(ll_dir: Path, candidate: str) -> dict[str, str]:
    """Per endpoint, the matrix template (default or nosys) under which its replies are likelier."""
    header, per_variant = load_scores(ll_dir / f"ll_{STAGE_MATRIX}_{candidate}.npz")
    endpoints = header["endpoint"]
    totals: dict[str, dict[str, float]] = {}
    for template in (TEMPLATE_DEFAULT, TEMPLATE_NOSYS):
        sums = row_sums(per_variant[(template, VARIANT_BASE)], "logp")
        for endpoint, value in zip(endpoints, sums, strict=True):
            totals.setdefault(str(endpoint), {}).setdefault(template, 0.0)
            totals[str(endpoint)][template] += float(value)
    return {e: max(t, key=lambda name: t[name]) for e, t in totals.items()}


ZOO_CANDIDATE: Mapping[str, str] = {
    "qwen15": "qwen2.5-1.5b", "qwen3": "qwen2.5-3b", "llama1": "llama3.2-1b", "smollm": "smollm2-360m",
}


def zoo_command(args: argparse.Namespace) -> int:
    """Full-feature validation on the zoo: separation and simulated official score per family."""
    zoo_dir, ll_root = Path(args.zoo_dir), Path(args.ll_dir)
    sep_rows: list[dict[str, Any]] = []
    sim_rows: list[dict[str, Any]] = []
    for family in args.families:
        candidate = ZOO_CANDIDATE[family]
        ll_dir = ll_root / family
        if not (ll_dir / f"ll_{STAGE_MATRIX}_{candidate}.npz").is_file():
            print("no containers for", family)
            continue
        class_of = load_endpoint_classes(zoo_dir / f"meta_{family}.json")
        templates = best_templates(ll_dir, candidate)
        table = build_calibration_table(ll_dir, candidate, templates, args.fp8_stages)
        budgets = attainable_budgets(table, class_of, args.budgets)
        print(family, "budgets", budgets, "llr features", sorted(table.llr), flush=True)
        for row in separation_table(table, budgets=budgets, draws=args.draws, class_of=class_of,
                                    clean_classes=ZOO_CLEAN_CLASSES, target_classes=ZOO_TARGET_CLASSES):
            sep_rows.append({"family": family, **row})
        for budget in budgets:
            for oracle in (True, False):
                for margin in args.margins:
                    for scale in (1.0, REAL_FP8_EFFECT_SCALE):
                        result = simulate_families(
                            table, class_of, budget, oracle_identity=oracle, fp8_feature=args.fp8_feature,
                            fp8_margin_sigma=margin, trials=args.trials, fp8_effect_scale=scale,
                            exclude=tuple(args.exclude),
                        )
                        sim_rows.append({
                            "source": family, "budget": budget, "oracle_identity": oracle,
                            "fp8_margin_sigma": margin, "fp8_effect_scale": scale,
                            "mean_score": round(result.mean_score, 3), "sd_score": round(result.sd_score, 3),
                            "median_score": round(result.median_score, 3),
                            "p10_score": round(result.p10_score, 3), "p90_score": round(result.p90_score, 3),
                            "isolated_share": json.dumps({k: round(v, 2) for k, v in result.isolated_share.items()}),
                        })
                        print(family, budget, "identity" if oracle else "no identity", "margin", margin,
                              "fp8 scale", scale, "median", round(result.median_score, 3),
                              "p10-p90", (round(result.p10_score, 3), round(result.p90_score, 3)), flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "zoo_separation.csv", ZOO_FULL_SEP_FIELDS, sep_rows)
    write_rows(out_dir / f"partition_sim_zoo{args.tag}.csv", PARTITION_SIM_FIELDS, sim_rows)
    return 0


TEMP_RULE_FIELDS: tuple[str, ...] = (
    "source", "budget", "min_temperature", "max_abs_z", "temp1_recall", "lora15_leak", "persona_leak",
    "system_leak", "plain_leak", "median_score", "p10_score", "p90_score",
)
TEMP_RULE_GRID: tuple[tuple[float | None, float], ...] = (
    (0.90, 2.5), (0.95, 2.0), (0.95, 1.5), (0.97, 1.5), (0.97, 1.0), (0.98, 1.0), (None, 1.5),
)
"""`(minimum fitted temperature, maximum |z at 1|)`; None = the sample-size-dependent bound."""


def pass_rate(
    draws: Mapping[str, Sequence[Mapping[str, float]]], klass: str, min_temperature: float, max_abs_z: float,
) -> float:
    """Share of the draws of a class that pass as temperature-1.0-like (NaN without draws)."""
    if klass not in draws:
        return float("nan")
    return float(np.mean([
        d["mle_temperature"] >= min_temperature and abs(d["z_at_one"]) <= max_abs_z for d in draws[klass]
    ]))


def temp_rule_command(args: argparse.Namespace) -> int:
    """How often each class passes as temperature-1.0-like, and the score, for a grid of thresholds."""
    identification = json.loads(Path(args.identification).read_text(encoding="utf-8"))
    templates = {row["endpoint"]: row["template"] for row in identification}
    ref_classes: Mapping[str, str] = CLASS_OF
    sources: list[tuple[str, RowFeatureTable, Mapping[str, str]]] = [
        ("ref", build_calibration_table(Path(args.ll_dir), "smollm2-360m", templates, args.fp8_stages), ref_classes)
    ]
    zoo_dir, zoo_ll = Path(args.zoo_dir), Path(args.zoo_ll_dir)
    for family in args.families:
        candidate = ZOO_CANDIDATE[family]
        ll_dir = zoo_ll / family
        if (ll_dir / f"ll_{STAGE_MATRIX}_{candidate}.npz").is_file():
            zoo_classes = load_endpoint_classes(zoo_dir / f"meta_{family}.json")
            zoo_table = build_calibration_table(
                ll_dir, candidate, best_templates(ll_dir, candidate), args.zoo_fp8_stages)
            sources.append((family, zoo_table, zoo_classes))
    rows: list[dict[str, Any]] = []
    for source, table, class_of in sources:
        endpoint_of_class = {klass: endpoint for endpoint, klass in class_of.items()}
        for budget in args.budgets:
            if source != "ref" and budget not in attainable_budgets(table, class_of, args.budgets):
                continue
            rng = np.random.default_rng(20261122 + budget)
            draws: dict[str, list[dict[str, float]]] = {
                klass: draw_features(table, endpoint_of_class[klass], budget, args.draws, rng)
                for klass in ("temp1", "lora_1.5", "persona", "system", "plain")
                if klass in endpoint_of_class
            }
            available = next(iter(draws.values()))[0]
            fp8_feature = next(
                name for name in (args.fp8_feature, "llr_fp8bf16_w_tensor") if name in available
            )
            for grid_minimum, max_z in TEMP_RULE_GRID:
                minimum = temperature_one_threshold(budget) if grid_minimum is None else grid_minimum
                result = simulate_families(
                    table, class_of, budget, oracle_identity=True, fp8_feature=fp8_feature,
                    fp8_margin_sigma=2.5, trials=args.trials, exclude=("fp8_w8a8",),
                    temperature_one_min=grid_minimum, temperature_one_max_abs_z=max_z,
                )
                rows.append({
                    "source": source, "budget": budget, "min_temperature": round(minimum, 4), "max_abs_z": max_z,
                    "temp1_recall": round(pass_rate(draws, "temp1", minimum, max_z), 3),
                    "lora15_leak": round(pass_rate(draws, "lora_1.5", minimum, max_z), 3),
                    "persona_leak": round(pass_rate(draws, "persona", minimum, max_z), 3),
                    "system_leak": round(pass_rate(draws, "system", minimum, max_z), 3),
                    "plain_leak": round(pass_rate(draws, "plain", minimum, max_z), 3), "median_score": round(result.median_score, 3),
                    "p10_score": round(result.p10_score, 3), "p90_score": round(result.p90_score, 3),
                })
                print({k: rows[-1][k] for k in ("source", "budget", "min_temperature", "max_abs_z", "temp1_recall",
                                                 "lora15_leak", "median_score")}, flush=True)
    write_rows(Path(args.out_dir) / "temp_rule_grid.csv", TEMP_RULE_FIELDS, rows)
    return 0


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", maxsplit=1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    matrix = sub.add_parser("matrix")
    matrix.add_argument("--ll-dir", required=True)
    matrix.add_argument("--answers-dir", required=True)
    matrix.add_argument("--out-dir", required=True)
    fp8 = sub.add_parser("fp8")
    fp8.add_argument("--ll-dir", required=True)
    fp8.add_argument("--identification", required=True)
    fp8.add_argument("--stage", default=STAGE_FP8)
    fp8.add_argument("--out-dir", required=True)
    calib = sub.add_parser("calib")
    calib.add_argument("--ll-dir", required=True)
    calib.add_argument("--identification", required=True)
    calib.add_argument("--out-dir", required=True)
    calib.add_argument("--candidate", default="smollm2-360m")
    calib.add_argument("--fp8-stages", nargs="*", default=["fp8", "fp8x", "fp8bf16"])
    calib.add_argument("--draws", type=int, default=400)
    realism = sub.add_parser("realism")
    realism.add_argument("--ll-dir", required=True)
    realism.add_argument("--identification", required=True)
    realism.add_argument("--out-dir", required=True)
    realism.add_argument("--candidate", default="smollm2-360m")
    realism.add_argument("--fp8-stages", nargs="*", default=["fp8", "fp8x", "fp8bf16"])
    zoo_ll = sub.add_parser("zoo-ll")
    zoo_ll.add_argument("--zoo-dir", required=True)
    zoo_ll.add_argument("--out-dir", required=True)
    psim = sub.add_parser("partition-sim")
    psim.add_argument("--ll-dir", required=True)
    psim.add_argument("--identification", required=True)
    psim.add_argument("--out-dir", required=True)
    psim.add_argument("--candidate", default="smollm2-360m")
    psim.add_argument("--fp8-stages", nargs="*", default=["fp8", "fp8x", "fp8bf16"])
    psim.add_argument("--fp8-feature", default="llr_fp8_w_tensor")
    psim.add_argument("--budgets", nargs="+", type=int, default=[560, 1500, 2600, 3500, 4400])
    psim.add_argument("--margins", nargs="+", type=float, default=[2.5])
    psim.add_argument("--trials", type=int, default=150)
    zoo = sub.add_parser("zoo")
    zoo.add_argument("--zoo-dir", required=True)
    zoo.add_argument("--ll-dir", required=True)
    zoo.add_argument("--out-dir", required=True)
    zoo.add_argument("--families", nargs="+", default=["qwen15", "qwen3", "llama1", "smollm"])
    zoo.add_argument("--fp8-stages", nargs="*", default=["fp8", "fp8bf16"])
    zoo.add_argument("--fp8-feature", default="llr_fp8_w_tensor")
    zoo.add_argument("--budgets", nargs="+", type=int, default=[560, 1500, 2600, 3500, 4400])
    zoo.add_argument("--margins", nargs="+", type=float, default=[1.0, 2.0, 2.5, 3.0])
    zoo.add_argument("--draws", type=int, default=200)
    zoo.add_argument("--trials", type=int, default=150)
    zoo.add_argument("--exclude", nargs="*", default=["fp8_w8a8"])
    zoo.add_argument("--tag", default="")
    trule = sub.add_parser("temp-rule")
    trule.add_argument("--ll-dir", required=True)
    trule.add_argument("--identification", required=True)
    trule.add_argument("--zoo-dir", required=True)
    trule.add_argument("--zoo-ll-dir", required=True)
    trule.add_argument("--out-dir", required=True)
    trule.add_argument("--families", nargs="+", default=["qwen15", "qwen3", "llama1"])
    trule.add_argument("--fp8-stages", nargs="*", default=["fp8", "fp8x", "fp8bf16"])
    trule.add_argument("--zoo-fp8-stages", nargs="*", default=["fp8bf16"])
    trule.add_argument("--fp8-feature", default="llr_fp8_w_tensor")
    trule.add_argument("--budgets", nargs="+", type=int, default=[560, 1500, 3500])
    trule.add_argument("--draws", type=int, default=300)
    trule.add_argument("--trials", type=int, default=100)
    sim = sub.add_parser("sim")
    sim.add_argument("--ll-dir", required=True)
    sim.add_argument("--out-dir", required=True)
    partition = sub.add_parser("partition")
    partition.add_argument("--data-dir", required=True)
    partition.add_argument("--answers-dir", required=True)
    partition.add_argument("--out-name", default="partition_v3.csv")
    classify_parser = sub.add_parser("classify")
    classify_parser.add_argument("--ll-dir", required=True)
    classify_parser.add_argument("--identification", required=True)
    classify_parser.add_argument("--out-dir", required=True)
    classify_parser.add_argument("--candidate", default="smollm2-360m")
    classify_parser.add_argument("--fp8-stages", nargs="*", default=["fp8", "fp8x", "fp8bf16"])
    classify_parser.add_argument("--budgets", nargs="+", type=int, default=[560, 1500, 4000])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    if args.command == "matrix":
        return matrix_command(args)
    if args.command == "fp8":
        return fp8_command(args)
    if args.command == "calib":
        return calib_command(args)
    if args.command == "classify":
        return classify_command(args)
    if args.command == "partition":
        return partition_command(args)
    if args.command == "sim":
        return sim_command(args)
    if args.command == "realism":
        return realism_command(args)
    if args.command == "zoo-ll":
        return zoo_ll_command(args)
    if args.command == "zoo":
        return zoo_command(args)
    if args.command == "temp-rule":
        return temp_rule_command(args)
    if args.command == "partition-sim":
        return partition_sim_command(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
