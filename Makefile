# Makefile wmkit — raccourcis pour la preparation du championnat.
# Usage : `make` (ou `make help`) liste les cibles. `make test` lance la suite rapide.
#
# Convention : les commandes lourdes (suite lente, eval complet, entrainement CIFAR)
# passent par `jarvis-heavy` quand il est present, pour partager proprement la machine.

PY      := .venv/bin/python
IMAGES  := data/kodak
IMAGE   := $(IMAGES)/kodim01.png
RESULTS := results
# jarvis-heavy si present, sinon chaine vide (la commande tourne alors en direct).
HEAVY   := $(shell command -v jarvis-heavy 2>/dev/null)

.DEFAULT_GOAL := help
.PHONY: help setup data weights list smoke demo demo-text demo-met demo-unlearn \
        eval test test-slow test-all lint fmt clean

help: ## Lister les cibles disponibles
	@echo "wmkit — cibles make :"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

# --- Installation et donnees ---------------------------------------------------

setup: ## Reconstruire l'environnement Python (.venv)
	./scripts/setup_env.sh

data: ## Telecharger les images de test (Kodak, 24 images)
	$(PY) scripts/fetch_data.py --dataset kodak

weights: ## Telecharger les poids pre-entraines (Pixel Seal / Video Seal)
	$(PY) scripts/fetch_weights.py

# --- Verifications rapides -----------------------------------------------------

list: ## Lister methodes, attaques et suites
	$(PY) -m wmkit list

smoke: ## Sanity check : import + liste (quelques secondes)
	@$(PY) -c "import wmkit; print('import OK')"
	@$(PY) -m wmkit list >/dev/null && echo "CLI OK"

# --- Demonstrations de bout en bout --------------------------------------------

demo: ## Demo image : tatoue kodim01 avec Pixel Seal, extrait, affiche bit_acc + p-value
	@test -f $(IMAGE) || $(MAKE) data
	@MSG=$$($(PY) -m wmkit embed --method pixelseal --in $(IMAGE) \
		--out /tmp/wm.png --message random --seed 42 2>/dev/null); \
	echo "message tatoue : $$MSG"; \
	echo "--- extraction ---"; \
	$(PY) -m wmkit extract --method pixelseal --in /tmp/wm.png --message $$MSG 2>/dev/null

demo-text: ## Demo texte : tatouage KGW/Gumbel et detection (p-value exacte)
	$(PY) scripts/demo_text_watermark.py

demo-met: ## Demo audit : Model Equality Testing sur un petit modele
	$(HEAVY) $(PY) scripts/demo_met.py

demo-unlearn: ## Demo unlearning : SCRUB sur CIFAR (lourd, passe par jarvis-heavy)
	$(HEAVY) $(PY) scripts/demo_scrub_cifar.py

# --- Evaluation ----------------------------------------------------------------

eval: ## Benchmark Pixel Seal sur la suite du papier (46 attaques, lourd)
	@test -f $(IMAGE) || $(MAKE) data
	@mkdir -p $(RESULTS)
	$(HEAVY) $(PY) -m wmkit eval --method pixelseal --images $(IMAGES) \
		--suite paper --out $(RESULTS)/pixelseal.csv

# --- Tests et qualite ----------------------------------------------------------

test: ## Suite de tests rapide (exclut les tests lents)
	$(PY) -m pytest -q

test-slow: ## Suite lente uniquement (poids, reseau, CIFAR ; via jarvis-heavy)
	$(HEAVY) $(PY) -m pytest -q -m slow

test-all: ## Toute la suite, rapide + lente (via jarvis-heavy)
	$(HEAVY) $(PY) -m pytest -q -m "slow or not slow"

lint: ## Linters : ruff + mypy strict
	$(PY) -m ruff check src tests
	$(PY) -m mypy src

fmt: ## Formater et corriger automatiquement (ruff)
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

clean: ## Nettoyer les artefacts (resultats, caches, images tatouees /tmp)
	rm -rf $(RESULTS) /tmp/wm.png /tmp/wm2.png
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -prune -exec rm -rf {} +

# --- Docker (portabilite / reproductibilite, PAS la vitesse sur Mac) -----------------
# Sur macOS, Docker tourne dans une VM Linux sans acces au GPU Apple (MPS) : le conteneur
# est CPU-only et plus lent que le .venv natif. Il sert a rejouer l'environnement chez un
# collegue ou sur le serveur Linux de la competition. Le chemin rapide est un host Linux +
# GPU NVIDIA (image CUDA). Poids et donnees se montent en volume, jamais copies dans l'image.

DOCKER_CPU_IMAGE  := wmkit:cpu
DOCKER_CUDA_IMAGE := wmkit:cuda
# Meta videoseal n'est pas versionne dans ce depot (third_party/ est ignore) : le build en a
# besoin, on le recupere au commit epingle si absent (setup_env.sh clone la tete de branche).
VIDEOSEAL_DIR     := third_party/videoseal
VIDEOSEAL_URL     := https://github.com/facebookresearch/videoseal
VIDEOSEAL_REF     := f12875a9b5e711e0c6e2dc3a22aca18231ce4a1a
# Volumes hote -> conteneur ; /tmp en tmpfs (caches torch ephemeres, ne remplit pas le disque de la VM).
# --user : ecrit les sorties sous l'uid de l'hote (sur un serveur Linux, evite les fichiers root).
# Chemins quotes : tolere un repertoire de projet contenant une espace.
DOCKER_VOLUMES    := --user $(shell id -u):$(shell id -g) \
                     -v "$(CURDIR)/data:/app/data" -v "$(CURDIR)/weights:/app/weights" \
                     -v "$(CURDIR)/results:/app/results" --tmpfs /tmp:mode=1777
# Arguments de wmkit pour docker-run, ex. : make docker-run ARGS="eval --method pixelseal ..."
ARGS              ?= list

.PHONY: docker-build docker-run docker-shell docker-test docker-cuda-build

$(VIDEOSEAL_DIR)/pyproject.toml:
	mkdir -p $(VIDEOSEAL_DIR)
	git -C $(VIDEOSEAL_DIR) init -q
	git -C $(VIDEOSEAL_DIR) remote add origin $(VIDEOSEAL_URL) || git -C $(VIDEOSEAL_DIR) remote set-url origin $(VIDEOSEAL_URL)
	git -C $(VIDEOSEAL_DIR) fetch --depth 1 origin $(VIDEOSEAL_REF)
	git -C $(VIDEOSEAL_DIR) checkout -q FETCH_HEAD

docker-build: $(VIDEOSEAL_DIR)/pyproject.toml ## Construire l'image CPU wmkit:cpu (via jarvis-heavy)
	$(HEAVY) docker build -f docker/Dockerfile -t $(DOCKER_CPU_IMAGE) .

docker-run: ## Lancer wmkit dans le conteneur CPU (ARGS="list" par defaut, donnees montees en volume)
	@mkdir -p data weights results
	docker run --rm $(DOCKER_VOLUMES) $(DOCKER_CPU_IMAGE) $(ARGS)

docker-shell: ## Shell interactif dans le conteneur CPU (donnees montees en volume)
	@mkdir -p data weights results
	docker run --rm -it $(DOCKER_VOLUMES) --entrypoint /bin/bash $(DOCKER_CPU_IMAGE)

docker-test: ## Suite de tests rapide DANS le conteneur CPU (via jarvis-heavy)
	$(HEAVY) docker run --rm --tmpfs /tmp:mode=1777 -e PYTEST_ADDOPTS="-p no:cacheprovider" --entrypoint make $(DOCKER_CPU_IMAGE) test PY=python

docker-cuda-build: $(VIDEOSEAL_DIR)/pyproject.toml ## Construire l'image CUDA wmkit:cuda (host Linux + NVIDIA uniquement ; via jarvis-heavy)
	$(HEAVY) docker build -f docker/Dockerfile.cuda -t $(DOCKER_CUDA_IMAGE) .

# --- Hackathon (European AI Championship Paris 2026) ---------------------------------------
# Chaine idee -> experience -> soumission : docs/hackathon/process.md.
# Aucune cible ci-dessous ne depense le budget ep-* ni ne soumet : `--yes` reste un geste
# explicite, tape a la main (python -m hackathon.lab submit RUN --yes).

HK_ENV := PYTHONPATH=$(CURDIR)/src
HK_LAB := $(HK_ENV) $(PY) -m hackathon.lab
HK_UNL := $(HK_ENV) $(PY) -m hackathon.unlearning
HK_MET := $(HK_ENV) $(PY) -m hackathon.model_equality

.PHONY: hk-test hk-board hk-login hk-sync hk-new hk-run hk-status hk-fetch hk-submit \
        hk-unl-fetch hk-unl-smoke met-budget met-plan met-validate

hk-test: ## Hackathon : tests du kit (rapides, aucun reseau)
	$(HK_ENV) $(PY) -m pytest -q tests/hackathon

hk-board: ## Hackathon : tableau des runs (terminal) + runs/board.html (TASK= pour filtrer)
	$(HK_LAB) board $(if $(TASK),--task $(TASK))

hk-login: ## Hackathon : ouvre la session ssh JURECA (code TOTP une seule fois)
	$(HK_LAB) jureca login

hk-sync: ## Hackathon : rsync du depot vers JURECA (code seulement)
	$(HK_LAB) jureca sync

hk-new: ## Hackathon : nouvelle idee. make hk-new TASK=unlearning IDEA="gold soft"
	@test -n "$(TASK)" -a -n "$(IDEA)" || { echo 'usage: make hk-new TASK=unlearning|model_equality|wmloc IDEA="nom de l idee"'; exit 2; }
	$(HK_LAB) new --task "$(TASK)" --idea "$(IDEA)"

hk-run: ## Hackathon : lance une experience. make hk-run SPEC=<id> [WHERE=local|jureca]
	@test -n "$(SPEC)" || { echo 'usage: make hk-run SPEC=<id ou fichier> [WHERE=local|jureca]'; exit 2; }
	$(HK_LAB) run "$(SPEC)" $(if $(WHERE),--where $(WHERE))

hk-status: ## Hackathon : etat des jobs JURECA en file ou en cours
	$(HK_LAB) jureca status

hk-fetch: ## Hackathon : rapatrie un run JURECA termine. make hk-fetch RUN=<run_id>
	@test -n "$(RUN)" || { echo 'usage: make hk-fetch RUN=<run_id ou group_id>'; exit 2; }
	$(HK_LAB) jureca fetch "$(RUN)"

hk-submit: ## Hackathon : VALIDE un run (essai a blanc, n'envoie rien). make hk-submit RUN=<run_id>
	@test -n "$(RUN)" || { echo 'usage: make hk-submit RUN=<run_id>'; exit 2; }
	@# la validation seule sort 2 (confirmation explicite absente) : pas une erreur ici
	$(HK_LAB) submit "$(RUN)"; code=$$?; test $$code -le 2

hk-unl-fetch: ## Hackathon unlearning : telecharge model.pt et forget_images.npz
	$(HK_UNL) fetch

hk-unl-smoke: ## Hackathon unlearning : essai rapide gold_soft (config smoke, < 2 min sur M5)
	$(HK_UNL) run --method gold_soft --config smoke

met-budget: ## Model equality : budget serveur par endpoint (GET /models, ne coute rien)
	$(HK_MET) budget

met-plan: ## Model equality : cout exact d'une phase, hors ligne. make met-plan PHASE=ep-1
	@test -n "$(PHASE)" || { echo 'usage: make met-plan PHASE=ref-1|ref-full|ep-1|ep-2|ep-3'; exit 2; }
	$(HK_MET) plan --phase "$(PHASE)"

met-validate: ## Model equality : valide un CSV de soumission. make met-validate FILE=<csv>
	@test -n "$(FILE)" || { echo 'usage: make met-validate FILE=<submission.csv>'; exit 2; }
	$(HK_MET) validate "$(FILE)"
