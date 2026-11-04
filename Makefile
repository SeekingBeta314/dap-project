# Everything runs through the project venv, so no `source` and no
# python-vs-python3 confusion. Start with `make setup`.

PY := .venv/bin/python
SEED ?= 0
CONFIG ?= smoke

.PHONY: setup test scene dry-run experiment clean-venv

setup: $(PY)

$(PY): requirements.txt
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	@touch $(PY)
	@echo "\nDone. Try: make test   or   make scene"

test: $(PY)
	$(PY) -m pytest -q

scene: $(PY)
	$(PY) scripts/show_scene.py --seed $(SEED)

dry-run: $(PY)
	$(PY) scripts/run_experiment.py $(CONFIG) --dry-run

experiment: $(PY)
	$(PY) scripts/run_experiment.py $(CONFIG)

clean-venv:
	rm -rf .venv
