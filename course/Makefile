# Build and verify the course.
#
#   make build    src/*.py  ->  notebooks/*.ipynb
#   make verify   run every notebook, in stub and in solution mode
#   make check    build then verify (what CI would run)
#   make clean    remove generated notebooks and caches

PYTHON ?= python3

.PHONY: build verify check clean lab

build:
	$(PYTHON) tools/build.py

verify:
	$(PYTHON) tools/verify.py

check: build verify

lab:
	$(PYTHON) -m jupyter lab notebooks

clean:
	rm -rf notebooks/*.ipynb
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name '.ipynb_checkpoints' -type d -prune -exec rm -rf {} +
