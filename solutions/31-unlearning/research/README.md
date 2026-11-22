# Unlearning research workspace

Every method version tested on task 31 during the event, with the campaign specs that produced each
run, the run registry, the local bench and the experiment lab that ran them. The submitted solution
is described in [`../README.md`](../README.md); its body uses the pinned method copy in
[`../trust_region_src/`](../trust_region_src/), which is this code at commit 52fa1d3 (V1 to V5).
This folder holds the final state of the same code, which adds V5.1, V_Dumb, V_Dumb2 and V_Fusion.

It is a self-contained root: the lab resolves `experiments/`, `runs/` and `jureca/` from here.
The hackathon packages come from `src/`; `wmkit` (proxy images, test fixtures) from the repository's
own `src/`. Values specific to the event are placeholders: the server URL comes from
`HACKATHON_SERVER_URL` (see the repository's `.env.example`), the Slurm account is `trainingXXXX`
in `src/hackathon/lab/constants.py` and the job scripts under `jureca/`.

## Layout

| Path | Content |
|---|---|
| `src/hackathon/unlearning/` | Methods (`trust_region` V1 to V4, `band_ascent` V5 and V5.1, `dumb`, `scrub`, `gold_soft`), local bench, step-0 diagnosis, synthetic anchor of V_Fusion, scoreboard |
| `src/hackathon/lab/` | Experiment lab: TOML spec, sweep (one variant per GPU, 4 at most), local or JURECA run, registry, board, guarded submission |
| `experiments/unlearning/*.toml` | One spec per campaign sweep; its header says what it tests and how to read it |
| `experiments/unlearning/*.md` | Hypothesis sheets (thresholds set before any run) and campaign reports |
| `experiments/unlearning/measurements/` | Measured targets (V2, V3), V5 families, bench-v4 re-bench |
| `experiments/registry.jsonl` | 192 runs (384 snapshots, merged by union): parameters, metrics, code commit, submission file and public score when sent |
| `scripts/` | Target builders (`unlearning_v2_targets.py`, `unlearning_v3_targets.py`, `unlearning_v5_families.py`), GPU setup and guard, proxy image download |
| `jureca/` | Lab job template and the cluster jobs of the unlearning task |
| `docs/hackathon/` | Method notes (`unlearning.md`) and the lab process (`process.md`) |
| `rapport/` | LaTeX report of the unlearning work (`rapport.tex`, built `rapport.pdf`) |
| `tests/hackathon/` | Unit tests of the lab and the unlearning package |

## Versions and their campaigns

| Version | Method | Specs | Campaign report |
|---|---|---|---|
| V1 | `trust_region`, KL radius | `tr-*`, `trust-region-kl-radius` | `trust-region-campaign.md` |
| V2 | per-image targets from views (`trust_target=views`) | `tr2-*`, `trust-region-views` | `trust-region-v2-campaign.md` |
| V3 | calibrated target, bench-v4 | `tr3-*`, `ref4-*`, `trust-region-calibrated` | `trust-region-v3-campaign.md` |
| V4 | anchor from the forget set, active constraint, conform SCRUB | `tr4-*`, `trust-region-anchored`, `trust-region-dual`, `scrub-*` | `trust-region-v4-campaign.md` |
| V5 | `band_ascent`, banded forgetting dosed out of fold | `band-ascent`, `band-ascent-step0`, `band-locality` | `trust-region-v5-campaign.md` |
| V5.1 | band in margin, KL hold, retain floor | `band-ascent-v51*`, `band-ascent-budget`, `band-ascent-s1` | `trust-region-v51-campaign.md` |
| V_Dumb | `dumb`, the simplest method | `v-dumb*` | `trust-region-vdumb-campaign.md` |
| V_Dumb2 | V_Dumb hinge inside conform SCRUB | `v-dumb2` | `v-dumb2.md` |
| V_Fusion | conform SCRUB anchored on synthetic images | `v-fusion-*` | `trust-region-vfusion-campaign.md` |

References and noise of the bench: `bench-baselines`, `bench-seed-noise`, `gold-soft-dose*`.

## Rerun a version

```bash
cd solutions/31-unlearning/research
make hk-unl-fetch                        # model.pt and forget_images.npz (needs HACKATHON_API_KEY)
PYTHONPATH=src:../../../src python3 scripts/fetch_data.py --dataset coco_val_subset   # proxy images
make hk-run SPEC=tr2-radius-best         # any spec id; WHERE=jureca for the cluster
make hk-board TASK=unlearning            # runs/board.html
make hk-test                             # unit tests
```

A run never sends anything; `make hk-submit RUN=<id>` only validates. Each registry record keeps
the commit of the original branch it ran at; that branch is not published, and this folder is its
final state. `tests/hackathon/test_hk_unlearning_campaign_specs.py` checks that every spec, for each
variant and both places, still gives a command and a config that this code accepts.
