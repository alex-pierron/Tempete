"""Comparison table of the Unlearning runs: `experiments/unlearning/SCOREBOARD.md`.

The lab board ranks every task by one metric. Here the registry's full-scale Unlearning runs
(those with `bench.*`, i.e. the whole forget set and an audit fold) are laid out side by side:
the settings that differ from the defaults, the bench score under each gold stand-in, the
retain on the audit fold and the locality of the forgetting. A second table averages the seeds
of each setting, so a difference can be read against the training noise. Runs are grouped by
bench protocol, newest first: scores of two protocols are never ranked together.

    python -m hackathon.unlearning.scoreboard          # writes the file and prints the summary

The numbers are the LOCAL bench (see bench.py), never the server score.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path

from hackathon.lab.registry import Registry, RunRecord
from hackathon.unlearning.alterations import ALTERATIONS
from hackathon.unlearning.configs import UnlearnConfig

TASK = "unlearning"
STATUS_DONE = "done"
SCORE_KEY = "bench.score_min"
CONFIG_PREFIX = "config."
NOT_SETTINGS = frozenset({"seed", "device", "proxy_root", "holdout_fraction"})
"""Config fields that do not define a setting (the seed is averaged over, the rest is fixed)."""
NONE_TEXT = "none"
"""Shown for a numeric field set to None although its default is a number (band_floor=None)."""
PROTOCOL_KEY = "bench.protocol_version"
UNVERSIONED_PROTOCOL = 3
"""Runs written before bench-v4 carry no protocol number: they are all bench-v3."""
STAND_INS = {3: ("s40", "s50", "s60"), 4: ("s40", "s50", "s60", "m30", "m20", "m15")}
"""Gold stand-ins of each protocol (bench-v4 adds the patch-masking family)."""
DEFAULT_REGISTRY = Path("experiments/registry.jsonl")
DEFAULT_RUNS = Path("runs")
DEFAULT_OUT = Path("experiments/unlearning/SCOREBOARD.md")
SHORT_NAMES = {
    "kl_radius": "ε",
    "target_temperature": "τ",
    "target_view_min_area": "aire",
    "target_views": "vues",
    "lr": "lr",
    "trainable_blocks": "blocs",
    "batch_size": "lot",
    "epochs": "époques",
    "warmup_fraction": "warmup",
    "retain_weight": "w_ret",
    "altered_weight": "w_alt",
    "weight_decay": "wd",
    "radius_check_every": "check",
    "radius_fraction": "ρ",
    "trust_lambda": "λ",
    "trust_hedge": "b",
    "altered_fraction": "copies",
    "trust_target": "cible",
    "anchor_folds": "plis",
    "anchor_weight": "w_ancre",
    "anchor_views": "vues_ancre",
    "radius_dual_lr": "η_dual",
    "retain_source": "rétention",
    "synthetic_pools": "lots",
    "band_release": "φ",
    "band_levels": "niveaux",
    "band_tolerance": "κ",
    "band_floor": "δ0",
    "band_sharpness": "β",
    "band_space": "espace",
    "band_hold_kl": "δ_KL",
    "cv_folds": "K",
    "cv_max_drift": "Λmax",
    "cv_min_agreement": "a_min",
    "cv_min_retain": "r_min",
    "cv_scale_exponent": "ν",
    "cv_check_every": "contrôle",
}
"""Display names, in display order (unknown fields come last, by name)."""
TRUST_REGION = "trust_region"
RADIUS_FRACTION = "radius_fraction"
TARGET_LABELS = {
    "tempered": "trust_region V1",
    "views": "trust_region V2",
    "calibrated": "trust_region V3",
}
"""trust_region is labelled by its target (`trust_target`), read from the run's own JSON: the
registry keeps numbers only. Without the run folder (another machine), the label stays
`trust_region`: regenerate the table where the runs are."""
ALWAYS_SHOWN = {
    "trust_region V1": ("kl_radius", "target_temperature"),
    "trust_region V2": ("kl_radius", "target_temperature", "target_view_min_area", "target_views"),
    "trust_region V3": (
        "kl_radius",
        "target_temperature",
        "target_view_min_area",
        "target_views",
        "trust_lambda",
        "trust_hedge",
    ),
    "trust_region": ("kl_radius", "target_temperature"),
    "gold_soft": ("target_temperature",),
    "band_ascent": ("band_release", "cv_max_drift"),
    "dumb": ("dumb_target_loss", "retain_weight"),
}
"""Settings shown even at their default value: the knobs that define the method's run."""


@dataclass(frozen=True)
class ScoreRow:
    """One finished full-scale run."""

    run_id: str
    spec_id: str
    method: str
    seed: int
    settings: tuple[tuple[str, float | str], ...]
    metrics: Mapping[str, float]
    sha: str

    def metric(self, name: str) -> float:
        """A metric of the run, NaN when absent."""
        return self.metrics.get(name, math.nan)

    @property
    def protocol(self) -> int:
        """Bench protocol number of the run's score."""
        return int(self.metrics.get(PROTOCOL_KEY, UNVERSIONED_PROTOCOL))

    @property
    def setting_key(self) -> tuple[str, tuple[tuple[str, float | str], ...]]:
        """What the seeds of one setting share."""
        return self.method, self.settings

    @property
    def transfer(self) -> float:
        """Worst altered-copy divergence relative to the exact images (1 = full transfer)."""
        exact = self.metric("bench.groups.forget_exact.kl_orig_sub")
        altered = [self.metric(f"bench.groups.forget_{name}.kl_orig_sub") for name in ALTERATIONS]
        if not exact > 0 or any(math.isnan(value) for value in altered):
            return math.nan
        return min(altered) / exact


def _defaults() -> dict[str, float]:
    """Numeric defaults of UnlearnConfig; NaN for optional numbers (None by default), so that any
    value a run sets for them (radius_fraction, forget_subset...) counts as a setting."""
    values: dict[str, float] = {}
    for field in fields(UnlearnConfig):
        value = getattr(UnlearnConfig(), field.name)
        if isinstance(value, bool | int | float):
            values[field.name] = float(value)
        elif value is None and field.name not in ("device", "proxy_root"):
            values[field.name] = math.nan
    return values


def _text_defaults() -> dict[str, str]:
    """Text config fields that define a setting (the registry keeps numbers only); the method
    and trust_region's target are in the label instead."""
    defaults = UnlearnConfig()
    return {
        field.name: value
        for field in fields(UnlearnConfig)
        if isinstance(value := getattr(defaults, field.name), str)
        and field.name not in ("method", "trust_target")
    }


def _run_config(record: RunRecord, runs_dir: Path) -> Mapping[str, object]:
    """The config of the run's own JSON, empty when the file is missing or unreadable."""
    sidecar = runs_dir / record.task / record.run_id / "candidate.json"
    try:
        config = json.loads(sidecar.read_text(encoding="utf-8"))["config"]
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    return config if isinstance(config, dict) else {}


def _text_value(config: Mapping[str, object], record: RunRecord, name: str) -> object:
    """A config field from the run's own JSON when present, else the sweep parameter."""
    return config.get(name, record.params.get(name))


def _method_of(config: Mapping[str, object], record: RunRecord) -> str:
    """config.method from the run's own JSON when present, else the sweep parameter.

    trust_region gets its target in the label (V1 tempered, V2 views, V3 calibrated), runs that
    predate `trust_target` being V1; without the run's JSON the label stays `trust_region`.
    """
    method = _text_value(config, record, "method")
    if method == TRUST_REGION and config:
        target = config.get("trust_target", "tempered")
        return TARGET_LABELS.get(target, TRUST_REGION) if isinstance(target, str) else TRUST_REGION
    return method if isinstance(method, str) else "?"


def rows_from_records(records: Iterable[RunRecord], runs_dir: Path) -> list[ScoreRow]:
    """Finished Unlearning runs that carry a full-scale bench score, best first."""
    defaults = _defaults()
    text_defaults = _text_defaults()
    rows: list[ScoreRow] = []
    for record in records:
        if record.task != TASK or record.status != STATUS_DONE or SCORE_KEY not in record.metrics:
            continue
        config = _run_config(record, runs_dir)
        method = _method_of(config, record)
        settings: list[tuple[str, float | str]] = []
        for name, default in defaults.items():
            if name in NOT_SETTINGS:
                continue
            value = record.metrics.get(CONFIG_PREFIX + name)
            if (
                value is None
                and not math.isnan(default)
                and name in config
                and config[name] is None
            ):
                # e.g. band_floor=None (V5 ablation): the registry keeps numbers only, so only the
                # run's own JSON tells it from the default; without the run folder it looks default
                settings.append((name, NONE_TEXT))
                continue
            shown = name in ALWAYS_SHOWN.get(method, ())
            if value is not None and (shown or not math.isclose(value, default, rel_tol=1e-9)):
                settings.append((name, value))
        for name, text_default in text_defaults.items():  # e.g. retain_source
            text = _text_value(config, record, name)
            if name not in NOT_SETTINGS and isinstance(text, str) and text != text_default:
                settings.append((name, text))
        if any(name == RADIUS_FRACTION for name, _ in settings):
            # the radius is radius_fraction x the target's distance: kl_radius is unused
            settings = [(name, value) for name, value in settings if name != "kl_radius"]
        rows.append(
            ScoreRow(
                run_id=record.run_id,
                spec_id=record.spec_id,
                method=method,
                seed=int(record.metrics.get(CONFIG_PREFIX + "seed", 0)),
                settings=tuple(sorted(settings, key=_display_order)),
                metrics=dict(record.metrics),
                sha=record.sha[:7] + ("*" if record.dirty else ""),
            )
        )
    return sorted(rows, key=lambda row: (-row.metric(SCORE_KEY), row.run_id))


def _display_order(setting: tuple[str, float | str]) -> tuple[int, str]:
    order = list(SHORT_NAMES)
    name = setting[0]
    return (order.index(name) if name in order else len(order), name)


def _number(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return "-"
    if value != 0 and abs(value) < 10 ** (-digits):
        return f"{value:.1e}"
    return f"{value:.{digits}f}"


def _setting(name: str, value: float | str) -> str:
    if isinstance(value, str):
        return f"{SHORT_NAMES.get(name, name)}={value}"
    shown = f"{value:g}" if not float(value).is_integer() else str(int(value))
    return f"{SHORT_NAMES.get(name, name)}={shown}"


def settings_text(row: ScoreRow) -> str:
    """`ε=0.6 τ=6`: the settings that differ from the defaults ('défaut' when none)."""
    return " ".join(_setting(name, value) for name, value in row.settings) or "défaut"


def _table(header: Sequence[str], body: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(cells) + " |" for cells in body]
    return lines


def _mean_std(values: Sequence[float]) -> str:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return "-"
    if len(clean) == 1:
        return _number(clean[0])
    return f"{_number(statistics.fmean(clean))} ± {_number(statistics.stdev(clean))}"


def _stand_ins(rows: Sequence[ScoreRow]) -> tuple[str, ...]:
    """Gold stand-ins of the rows' protocol (rows of one protocol only)."""
    return STAND_INS.get(rows[0].protocol, STAND_INS[UNVERSIONED_PROTOCOL]) if rows else ()


def _per_seed(group: Sequence[ScoreRow], name: str) -> list[float]:
    """One value of metric `name` per seed: a replay of a seed (same setting, same seed: `ref4-*`,
    a run rerun on a clean tree) counts once. Clean runs win over `dirty` ones (sha ending in
    `*`); several runs of one seed are averaged (identical when replayed on the same machine)."""
    by_seed: dict[int, list[ScoreRow]] = {}
    for row in group:
        by_seed.setdefault(row.seed, []).append(row)
    values = []
    for seed in sorted(by_seed):
        runs = by_seed[seed]
        kept = [row for row in runs if not row.sha.endswith("*")] or runs
        values.append(statistics.fmean(row.metric(name) for row in kept))
    return values


def summary_lines(rows: Sequence[ScoreRow]) -> list[str]:
    """One line per setting (method + non-default settings), seeds averaged, best mean first.

    `rows` must share one bench protocol (`render` splits them). Each seed counts once
    (`_per_seed`): the `graines` column is the number of distinct seeds.
    """
    groups: dict[tuple[str, tuple[tuple[str, float], ...]], list[ScoreRow]] = {}
    for row in rows:
        groups.setdefault(row.setting_key, []).append(row)

    def mean_score(group: list[ScoreRow]) -> float:
        return statistics.fmean(_per_seed(group, SCORE_KEY))

    ordered = sorted(groups.values(), key=lambda group: -mean_score(group))
    stand_ins = _stand_ins(rows)
    body = []
    for rank, group in enumerate(ordered, start=1):
        first = group[0]
        body.append(
            [
                str(rank),
                first.method,
                settings_text(first),
                str(len({row.seed for row in group})),
                _mean_std(_per_seed(group, SCORE_KEY)),
                *(_mean_std(_per_seed(group, f"bench.score.{s}")) for s in stand_ins),
                _mean_std(_per_seed(group, "bench.retain")),
                _mean_std(_per_seed(group, "bench.movement_ratio")),
            ]
        )
    header = ["#", "méthode", "réglages", "graines", "score_min", *stand_ins]
    return _table([*header, "retain", "ratio"], body)


def runs_lines(rows: Sequence[ScoreRow]) -> list[str]:
    """Every run, best score first (rows of one bench protocol)."""
    stand_ins = _stand_ins(rows)
    body = [
        [
            row.run_id,
            row.method,
            settings_text(row),
            str(row.seed),
            _number(row.metric(SCORE_KEY)),
            *(_number(row.metric(f"bench.score.{s}")) for s in stand_ins),
            _number(row.metric("bench.retain")),
            _number(row.metric("bench.audit_agreement")),
            _number(row.metric("bench.audit_kl")),
            _number(row.metric("bench.groups.forget_exact.kl_orig_sub")),
            _number(row.metric("bench.movement_ratio"), 2),
            _number(row.transfer, 2),
            _number(row.metric("final.steps"), 0),
            _number(row.metric("train_seconds"), 0),
            row.sha,
        ]
        for row in rows
    ]
    header = ["run", "méthode", "réglages", "graine", "score_min", *stand_ins, "retain"]
    header += ["accord audit", "KL audit", "KL oubli", "ratio", "transfert", "pas", "s", "sha"]
    return _table(header, body)


def render(rows: Sequence[ScoreRow], generated_at: datetime) -> str:
    """The whole Markdown page."""
    lines = [
        "# Unlearning : tableau comparatif des runs",
        "",
        (
            f"Généré le {generated_at:%Y-%m-%d %H:%M} par `make hk-unl-scores` depuis "
            "`experiments/registry.jsonl` (ne pas éditer à la main). Banc local "
            "(`docs/hackathon/unlearning.md` §8), **pas** le score du serveur. Une section par "
            "protocole du banc, le plus récent d'abord : deux protocoles ne se comparent pas."
        ),
        "",
        (
            "- `score_min` : score aux formules du sujet sous la pire doublure du gold ; "
            "identité = 0, doublure atteinte = 1. `bench-v3` : trois doublures (`s40`, `s50`, "
            "`s60` : crops de 40, 50 et 60 % de l'aire) ; `bench-v4` : six, les mêmes et trois "
            "masquages de patchs (`m30`, `m20`, `m15` : 30, 20 et 15 % des patchs)."
        ),
        (
            "- `retain` : accord × 1/(1 + KL) sur le fold d'audit (20 % du forget set, jamais "
            "entraîné) ; porte à 70 % d'accord."
        ),
        (
            "- `ratio` : divergence moyenne sur les images oubliées et leurs copies / celle du fold "
            "d'audit (grand = oubli localisé). `transfert` : pire copie altérée / images exactes. "
            "`KL oubli` : KL(orig‖sub) à T = 2 sur les images exactes entraînées."
        ),
        (
            "- `réglages` : ce qui diffère des défauts de `UnlearnConfig` (ε et τ toujours montrés "
            "pour `trust_region`). Une différence ne compte que si elle dépasse l'écart entre "
            "graines d'un même réglage. Dans la synthèse, chaque graine compte une fois : un "
            "rejeu (`ref4-*`, run relancé sur un arbre propre) ne double pas l'effectif, et un run "
            "propre l'emporte sur un run `dirty` (sha suivi de `*`)."
        ),
        "",
    ]
    for protocol in sorted({row.protocol for row in rows}, reverse=True):
        group = [row for row in rows if row.protocol == protocol]
        lines += [
            f"## bench-v{protocol} : synthèse par réglage ({len(group)} runs)",
            "",
            *summary_lines(group),
            "",
            f"## bench-v{protocol} : tous les runs",
            "",
            *runs_lines(group),
            "",
        ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Write the scoreboard and print its summary table."""
    parser = argparse.ArgumentParser(prog="python -m hackathon.unlearning.scoreboard")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS, help="local run folders")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    rows = rows_from_records(Registry(args.registry).latest().values(), args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(rows, datetime.now()), encoding="utf-8")
    newest = max((row.protocol for row in rows), default=UNVERSIONED_PROTOCOL)
    print("\n".join(summary_lines([row for row in rows if row.protocol == newest])))
    print(f"\n{len(rows)} run(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
