.PHONY: help venv figures clean

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help:
	@echo "Available targets:"
	@echo "  make venv      - create local venv and install dev deps (matplotlib, numpy)"
	@echo "  make figures   - regenerate figures/*.png from results/"
	@echo "  make clean     - remove figures/"

venv:
	python3 -m venv $(VENV)
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -r requirements-dev.txt
	@echo "venv ready. Activate with: source $(VENV)/bin/activate"

figures: $(VENV)
	$(PY) scripts/make_figures.py

$(VENV):
	$(MAKE) venv

clean:
	rm -f figures/*.png
