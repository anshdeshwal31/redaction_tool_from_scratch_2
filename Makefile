# Developer and CI entry points (plan Appendix B).  Run inside the dev container or any shell with uv.
.DEFAULT_GOAL := help
UV ?= uv
PYTEST := $(UV) run pytest
LICENCE_DENY := GPL;SSPL;BUSL;Non-Commercial;NonCommercial;BY-NC;Commons Clause

.PHONY: help sync models selfcheck lint test test-slow test-all determinism leaktest verify-mutations licences ci

help:  ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

sync:  ## install the locked environment
	$(UV) sync --locked

models:  ## download and SHA-256-verify every pinned model into .models/
	$(UV) run mlredact models fetch --profile broad

selfcheck:  ## validate config, runtime profile, resources and models
	$(UV) run mlredact selfcheck

lint:  ## ruff (lint + format check) and mypy --strict
	$(UV) run ruff check src tests tools
	$(UV) run ruff format --check src tests tools
	$(UV) run mypy src

test:  ## fast suite: unit, property, security, determinism (no models)
	$(PYTEST) -m "not slow"

test-slow:  ## end-to-end and model-backed tests (needs `make models`)
	$(PYTEST) -m slow

test-all: test test-slow  ## everything

determinism:  ## same input -> byte-identical output (hash seeds, reruns, worker counts)
	$(PYTEST) tests/determinism
	$(PYTEST) -m slow tests/integration -k "identical or independent or deterministic or consistent or windowing"

leaktest:  ## no document text in logs, exceptions, manifests or run records
	$(PYTEST) tests/security/test_logging_safety.py
	$(PYTEST) -m slow tests/integration -k "manifest_contains_no_document_text or non_pdf"

verify-mutations:  ## deliberately broken outputs must fail V1-V6
	$(PYTEST) tests/security/test_verifier_mutations.py

licences:  ## licence gate: no copyleft (GPL family), source-available or non-commercial dependencies
	$(UV) run --with pip-licenses pip-licenses --partial-match --fail-on="$(LICENCE_DENY)" --ignore-packages mlredact

ci: lint licences test  ## what every pull request runs
