.PHONY: help dev test test-unit test-integration lint format db-start db-stop db-reset studio install e2e fresh fresh-seed down

PY := PYTHONPATH=. uv run

help:
	@echo "make install          - uv sync"
	@echo "make dev              - start nemara-backend in dev mode (auto-starts supabase)"
	@echo "make test             - run unit + integration tests"
	@echo "make test-unit        - run unit tests only"
	@echo "make test-integration - run integration tests only"
	@echo "make lint             - ruff + mypy"
	@echo "make format           - ruff format + fix"
	@echo "make db-start         - supabase start"
	@echo "make db-stop          - supabase stop"
	@echo "make db-reset         - wipe + reapply supabase/schema.sql + seed"
	@echo "make studio           - open Supabase Studio"
	@echo "make e2e              - scenario CLI for iOS verification (pass args via ARGS=...)"
	@echo "make fresh            - reset + bring up supabase + backend + ngrok (no seed)"
	@echo "make fresh-seed       - same as fresh + seed demo data"
	@echo "make down             - stop supabase + backend + ngrok"

install:
	uv sync
	@# macOS: Bash sandbox creates .venv with UF_HIDDEN set, which causes
	@# Python's site.py to skip editable .pth files. Strip the flag.
	@if [ "$$(uname)" = "Darwin" ]; then \
		chflags -R nohidden .venv 2>/dev/null || true; \
	fi

dev: db-start
	$(PY) uvicorn src.main:app --reload --host 0.0.0.0 --port 8002

test:
	$(PY) pytest test/ -v

test-unit:
	$(PY) pytest test/unit -v

test-integration:
	$(PY) pytest test/integration -v

lint:
	$(PY) ruff check src test
	$(PY) mypy src

format:
	$(PY) ruff format src test
	$(PY) ruff check --fix src test

db-start:
	@supabase status >/dev/null 2>&1 || supabase start

db-stop:
	supabase stop

db-reset:
	supabase db reset
	$(PY) python scripts/seed.py

studio:
	open http://localhost:54343

# Scenario CLI used to verify the iOS app end-to-end against the live
# local stack. Usage:
#   make e2e ARGS="reset"
#   make e2e ARGS="fire-event --patient maggie --kind fall --location 'Kitchen'"
e2e:
	$(PY) e2e $(ARGS)

# Full clean restart of the local stack — kills uvicorn, wipes Supabase
# volumes (storage + db), re-applies schema, recreates storage buckets.
# `make dev` is still a separate step (runs the backend in foreground).
fresh:
	./scripts/local-start.sh

fresh-seed:
	./scripts/local-start.sh --seed

down:
	./scripts/local-start.sh --down
