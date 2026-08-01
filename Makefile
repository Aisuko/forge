# Forge — the commands you actually run, in one place.
#
#   make            # this help
#   make train      # train, score, and name a champion char model
#   make check      # the same gate the pre-commit hook runs
#   make council    # build the council page and serve it on :8080
#
# Every target is a thin wrapper around a script in scripts/, which stays the
# single source of truth. Nothing here reimplements a build step; if a recipe
# needs more than one line, it belongs in a script instead.
#
# Knobs (override on the command line, e.g. `make council PORT=9000`):
#   PORT      port for `make council` / `make surprise`       [8080]
#   BACKEND   wgpu | cpu, passed to the trainer and scorer    [wgpu]
#   CKPT      checkpoint promoted by `make ship`
#   STEPS PATIENCE RUNS OUT   forwarded to scripts/train_char.sh

.DEFAULT_GOAL := help
SHELL := /usr/bin/env bash

PORT    ?= 8080
BACKEND ?= wgpu
CKPT    ?= checkpoints/shakespeare_char.safetensors

# `make train STEPS=600` works because the script reads these from the
# environment; make only exports what it is told to.
export BACKEND
export STEPS
export PATIENCE
export RUNS
export OUT

.PHONY: help train train-quick train-baseline train-report ship \
        council surprise top check check-full test test-parity fmt clippy \
        data gpt2 hooks clean

help: ## Show this help
	@printf '\033[1mForge\033[0m — make <target>\n\n'
	@grep -hE '^[a-z][a-zA-Z0-9_-]*:.*?## ' $(MAKEFILE_LIST) \
	  | sort \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@printf '\nknobs: PORT=%s BACKEND=%s CKPT=%s\n' '$(PORT)' '$(BACKEND)' '$(CKPT)'

# ── training ─────────────────────────────────────────────────────────────────
# scripts/train_char.sh scores what already exists, trains several recipes,
# ranks them, and prints the command to ship the winner. Safe to interrupt.

train: ## Full training campaign: train, score, name a champion (~10h)
	./scripts/train_char.sh

train-quick: ## 600-step smoke test of the whole training flow
	./scripts/train_char.sh --quick

train-baseline: ## Score the checkpoints you already have, train nothing (~4min)
	./scripts/train_char.sh --baseline-only

train-report: ## Re-render the campaign report from existing scores
	./scripts/train_char.sh --report

ship: ## Promote a checkpoint into assets/ (CKPT=path/to.safetensors)
	./scripts/ship_char_model.sh $(CKPT)
	@printf '\nnow regenerate the browser gate fixture:\n'
	@printf '  cargo run --release --example gate_tokens -- --model assets/shakespeare_char\n'

# ── tools ────────────────────────────────────────────────────────────────────

council: ## Build the council page and serve it on :$(PORT)
	./tools/council/build.sh
	python3 -m http.server -d tools/council/dist --bind 0.0.0.0 $(PORT)

surprise: ## Build the Surprise page and serve it on :$(PORT)
	./tools/surprise/build.sh
	python3 -m http.server -d tools/surprise/dist --bind 0.0.0.0 $(PORT)

top: ## Run forge-top, the terminal model browser
	cargo run --release -p forge-top -- --path models/ --path checkpoints/

# ── testing ──────────────────────────────────────────────────────────────────
# scripts/ci_local.sh defines what "green" means; the git hooks call the same
# two stages. `check` is pre-commit, `check-full` is pre-push.

check: ## Fast gate: fmt, clippy, wasm + tool builds, dep assert (~9s)
	./scripts/ci_local.sh fast

check-full: ## Everything in check, plus the release test suite (~1m10s)
	./scripts/ci_local.sh full

# `--features train`: it is off by default, so without it cargo silently skips
# tests/training.rs, tests/autograd.rs and tests/train_ops.rs. Matches
# scripts/ci_local.sh.
test: ## Run the release test suite only
	cargo test -p forge-ml --release --locked --features train
	cargo test -p forge-council --release --locked

test-parity: ## The numeric parity suites against the PyTorch goldens
	cargo test --release --test op_parity
	cargo test --release --test gpt2_e2e -- golden

fmt: ## Format the workspace
	cargo fmt --all

clippy: ## Lint with warnings denied
	cargo clippy -p forge-ml --all-targets --locked -- -D warnings
	cargo clippy -p forge-ml --all-targets --locked --features train -- -D warnings
	cargo clippy -p forge-council -p forge-top --all-targets --locked -- -D warnings

# ── setup ────────────────────────────────────────────────────────────────────

data: ## Fetch the tinyshakespeare corpus
	./scripts/download_shakespeare.sh

gpt2: ## Fetch GPT-2 weights (needed by gpt2_e2e and kv_cache)
	./scripts/download_gpt2.sh

hooks: ## Install the pre-commit / pre-push hooks
	./scripts/install_hooks.sh

clean: ## Remove the built tool pages
	rm -rf tools/council/dist tools/surprise/dist
