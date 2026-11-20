# Everything runs through the project venv, so no `source` and no
# python-vs-python3 confusion. Start with `make setup`.

PY := .venv/bin/python
DEPS := .venv/.deps-installed
SEED ?= 0
CONFIG ?= smoke
SCENARIO ?= nominal
# Drake wheels need Python 3.12+; plain `python3` may be an older one (e.g.
# Anaconda's 3.11). Prefer 3.13, then 3.12. Override with `make setup PYTHON=...`.
PYTHON ?= $(shell command -v python3.13 || command -v python3.12 || command -v python3)

.PHONY: setup test scene dry-run experiment bank clean-venv

setup: $(DEPS)

$(DEPS): requirements.txt
	@$(PYTHON) -c 'import sys; assert sys.version_info >= (3, 12), f"need Python 3.12+ for Drake, got {sys.version.split()[0]} ($(PYTHON)); try make setup PYTHON=/path/to/python3.13"'
	@test -x $(PY) || $(PYTHON) -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	@touch $@
	@echo "\nDone. Try: make test   or   make scene"

test: $(DEPS)
	$(PY) -m pytest -q

scene: $(DEPS)
	$(PY) scripts/show_scene.py --seed $(SEED) --scenario $(SCENARIO)

dry-run: $(DEPS)
	$(PY) scripts/run_experiment.py $(CONFIG) --dry-run

experiment: $(DEPS)
	$(PY) scripts/run_experiment.py $(CONFIG)

N ?= 20000
WORKERS ?= 8

bank: $(DEPS)
	$(PY) scripts/build_bank.py --n $(N) --preset $(SCENARIO) --workers $(WORKERS)

clean-venv:
	rm -rf .venv
