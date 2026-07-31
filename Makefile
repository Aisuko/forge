# Forge — the commands you actually run, in one place.
#
#   make            # this help
#   make train      # train, score, and name a champion char model
#   make site serve # build the explainer site and serve it on :8080
#   make check      # the same gate the pre-commit hook runs
#
# Every target is a thin wrapper around a script in scripts/, which stays the
# single source of truth. Nothing here reimplements a build step; if a recipe
# needs more than one line, it belongs in a script instead.
#
# Knobs (override on the command line, e.g. `make serve PORT=9000`):
#   PORT      port for `make serve`                          [8080]
#   BACKEND   wgpu | cpu, passed to the trainer and scorer    [wgpu]
#   CKPT      checkpoint promoted by `make ship`
#   STEPS PATIENCE RUNS OUT   forwarded to scripts/train_char.sh

.DEFAULT_GOAL := help
SHELL := /usr/bin/env bash

PORT    ?= 8080
BACKEND ?= wgpu
DIST    := docs/dist
CKPT    ?= checkpoints/shakespeare_char.safetensors

# `make train STEPS=600` works because the script reads these from the
# environment; make only exports what it is told to.
export BACKEND
export STEPS
export PATIENCE
export RUNS
export OUT

.PHONY: help train train-quick train-baseline train-report ship \
        site site-fast serve check check-full test test-parity fmt clippy \
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
	@printf '\nnow regenerate the browser gate fixture and rebuild:\n'
	@printf '  cargo run --release --example gate_tokens -- --model assets/shakespeare_char\n'
	@printf '  make site\n'

# ── site ─────────────────────────────────────────────────────────────────────

site: ## Build the explainer site into docs/dist/ (includes wasm)
	./scripts/build_site.sh

site-fast: ## Build the site without wasm — fast, but the demo will 404
	./scripts/build_site.sh --no-wasm

serve: ## Serve docs/dist/ at http://localhost:$(PORT)/
	@[[ -f $(DIST)/index.html ]] || { echo "run 'make site' first" >&2; exit 1; }
	python3 -m http.server -d $(DIST) --bind 0.0.0.0 $(PORT)

# ── testing ──────────────────────────────────────────────────────────────────
# scripts/ci_local.sh defines what "green" means; the git hooks call the same
# two stages. `check` is pre-commit, `check-full` is pre-push.

check: ## Fast gate: fmt, clippy, wasm + tui builds, dep-leak assert (~6s)
	./scripts/ci_local.sh fast

check-full: ## Everything in check, plus the release test suite (~1m10s)
	./scripts/ci_local.sh full

# `--features council` on both: the feature is off by default, so without it
# cargo silently skips tests/council.rs and clippy never sees the module. These
# match what scripts/ci_local.sh runs.
test: ## Run the release test suite only
	cargo test --release --locked --features council

test-parity: ## The numeric parity suites against the PyTorch goldens
	cargo test --release --test op_parity
	cargo test --release --test gpt2_e2e -- golden

fmt: ## Format the workspace
	cargo fmt --all

clippy: ## Lint with warnings denied
	cargo clippy --all-targets --locked -- -D warnings
	cargo clippy --all-targets --locked --features council -- -D warnings

# ── setup ────────────────────────────────────────────────────────────────────

data: ## Fetch the tinyshakespeare corpus
	./scripts/download_shakespeare.sh

gpt2: ## Fetch GPT-2 weights (needed by gpt2_e2e and kv_cache)
	./scripts/download_gpt2.sh

hooks: ## Install the pre-commit / pre-push hooks
	./scripts/install_hooks.sh

clean: ## Remove the built site
	rm -rf $(DIST)
