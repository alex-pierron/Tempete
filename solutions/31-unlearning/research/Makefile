# Unlearning research workspace: the hk-* targets of the original team Makefile, run from this folder.
# The hackathon packages come from ./src; wmkit (proxy images, test fixtures) from the repository src/.
# Heavy local runs go through scripts/gpu_guard.sh on a machine with an NVIDIA GPU, directly elsewhere.

PY      := $(or $(wildcard ../../../.venv/bin/python),python3)
HEAVY   := $(if $(shell command -v nvidia-smi 2>/dev/null),./scripts/gpu_guard.sh)
HK_ENV  := PYTHONPATH=$(CURDIR)/src:$(CURDIR)/../../../src
HK_LAB  := $(HK_ENV) $(PY) -m hackathon.lab
HK_UNL  := $(HK_ENV) $(PY) -m hackathon.unlearning

.DEFAULT_GOAL := help
.PHONY: help hk-test hk-board hk-login hk-sync hk-new hk-run hk-status hk-fetch hk-submit \
        hk-unl-fetch hk-unl-scores hk-unl-smoke

help: ## List the targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- Hackathon: lab and unlearning shortcuts (never --yes, never query-ep) ---

hk-test: ## Unit tests of the lab and the unlearning package (fast, no network)
	$(PY) -m pytest -q

hk-board: ## Table of the runs (terminal) + runs/board.html (TASK= to filter)
	$(HK_LAB) board $(if $(TASK),--task $(TASK))

hk-login: ## Open the JURECA ssh session (TOTP code once)
	$(HK_LAB) jureca login

hk-sync: ## rsync of the code to JURECA
	$(HK_LAB) jureca sync

hk-new: ## New idea. make hk-new TASK=unlearning IDEA="gold soft"
	@test -n "$(TASK)" -a -n "$(IDEA)" || { echo 'usage: make hk-new TASK=unlearning IDEA="idea name"'; exit 2; }
	$(HK_LAB) new --task "$(TASK)" --idea "$(IDEA)"

hk-run: ## Run an experiment. make hk-run SPEC=<id> [WHERE=local|jureca]
	@test -n "$(SPEC)" || { echo 'usage: make hk-run SPEC=<id or file> [WHERE=local|jureca]'; exit 2; }
	$(HK_ENV) $(if $(filter jureca,$(WHERE)),,$(HEAVY)) $(PY) -m hackathon.lab run "$(SPEC)" $(if $(WHERE),--where $(WHERE))

hk-status: ## JURECA jobs queued or running
	$(HK_LAB) jureca status

hk-fetch: ## Bring back a finished JURECA run. make hk-fetch RUN=<run_id>
	@test -n "$(RUN)" || { echo 'usage: make hk-fetch RUN=<run_id or group_id>'; exit 2; }
	$(HK_LAB) jureca fetch "$(RUN)"

hk-submit: ## VALIDATE a run (dry run, sends nothing). make hk-submit RUN=<run_id>
	@test -n "$(RUN)" || { echo 'usage: make hk-submit RUN=<run_id>'; exit 2; }
	@# validation alone exits 2 (explicit confirmation missing): not an error here
	$(HK_LAB) submit "$(RUN)"; code=$$?; test $$code -le 2

hk-unl-fetch: ## Download model.pt and forget_images.npz
	$(HK_UNL) fetch

hk-unl-scores: ## Comparison table of the runs -> experiments/unlearning/SCOREBOARD.md
	$(HK_UNL).scoreboard

hk-unl-smoke: ## Quick gold_soft run (smoke config, a few minutes on a laptop)
	$(HK_ENV) $(HEAVY) $(PY) -m hackathon.unlearning run --method gold_soft --config smoke
