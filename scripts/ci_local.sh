#!/usr/bin/env bash
# The local CI gate. This script is the single source of truth for what
# "green" means: the hooks in .githooks/ are thin wrappers around it, and
# .github/workflows/ci.yml (manual dispatch only) runs the same stages.
#
#   ./scripts/ci_local.sh fast   # fmt, clippy, the builds, dep-leak assert
#   ./scripts/ci_local.sh full   # everything in fast, plus the release tests
#
# Every optional feature (`council`, `train`, `tui`) gets its own explicit run. A
# feature nobody builds is a feature that rots.
#
# `fast` runs on pre-commit (~6s with a warm target/), `full` on pre-push
# (~1m10s). Bypass either with `git commit --no-verify` / `git push
# --no-verify`.
#
# Stages are ordered cheapest-first and the script stops at the first failure,
# so a formatting slip does not cost you a minute of GPU tests.
set -uo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:-fast}"
case "$STAGE" in
  fast|full) ;;
  *) echo "usage: $0 [fast|full]" >&2; exit 2 ;;
esac

if [[ -t 1 ]]; then
  BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'; DIM=$'\033[2m'; OFF=$'\033[0m'
else
  BOLD=''; RED=''; GREEN=''; DIM=''; OFF=''
fi

step=0
started=$SECONDS

run() {
  local name="$1"; shift
  step=$((step + 1))
  printf '%s[%d] %s%s\n' "$BOLD" "$step" "$name" "$OFF"
  local t0=$SECONDS
  if ! "$@"; then
    printf '\n%s%s✗ %s failed%s %s(%s)%s\n' "$RED" "$BOLD" "$name" "$OFF" "$DIM" "$*" "$OFF"
    printf '%sfix it, or bypass this gate with --no-verify%s\n' "$DIM" "$OFF"
    exit 1
  fi
  printf '    %sok%s %s(%ss)%s\n' "$GREEN" "$OFF" "$DIM" "$((SECONDS - t0))" "$OFF"
}

# The TUI deps must never reach the library's dependents or the wasm build;
# they are optional, behind the `tui` feature, for exactly that reason.
assert_no_tui_deps() {
  local tree dep n
  tree=$(cargo tree -e normal --locked) || return 1
  for dep in ratatui crossterm sysinfo nvml-wrapper memmap2; do
    n=$(grep -c "^.*[^a-z-]$dep v" <<<"$tree" || true)
    if [[ "$n" -ne 0 ]]; then
      echo "$dep leaked into the default dependency tree" >&2
      return 1
    fi
  done
}

printf '%sforge local CI — %s stage%s\n\n' "$BOLD" "$STAGE" "$OFF"

run "cargo fmt --check"            cargo fmt --all --check
run "cargo clippy -D warnings"     cargo clippy --all-targets --locked -- -D warnings
# `council` is off by default, so the pass above walks straight past
# src/models/council.rs, its wasm bindings, its test and its example. Code that
# stops being linted the day it is feature-gated is worse off than code that was
# never gated, so lint it explicitly. `--features council` and not
# `--all-features`: the latter would pull the five TUI deps into every lint pass,
# and keeping them out of the default tree is what the check below is about.
run "clippy --features council"    cargo clippy --all-targets --locked --features council -- -D warnings
# Same argument for `train`: autograd, optim, the nine backward kernels and four
# test/example targets are invisible to the pass above. No matching wasm build,
# though — nothing on the site turns `train` on, and the default wasm build
# below is what proves it compiles *out*.
run "clippy --features train"      cargo clippy --all-targets --locked --features train -- -D warnings
# Two wasm builds for two different claims: the default one proves the council
# compiles *out* (what a dependent gets), the second that it compiles *in* (what
# scripts/build_web.sh ships to the site).
run "build wasm32"                 cargo build --release --locked --target wasm32-unknown-unknown
run "build wasm32 (council)"       cargo build --release --locked --features council --target wasm32-unknown-unknown
run "build forge-top (tui)"        cargo build --release --locked --features tui --bin forge-top
run "no TUI deps in default tree"  assert_no_tui_deps

if [[ "$STAGE" == full ]]; then
  # The weight-dependent suites (gpt2_e2e, kv_cache, the tokenizer's BPE cases)
  # self-skip rather than fail when models/gpt2/ is absent, which would make a
  # green run quietly weaker than it looks. Say so.
  if [[ ! -f models/gpt2/model.safetensors ]]; then
    printf '%snote: models/gpt2/ missing — gpt2_e2e and kv_cache will self-skip%s\n' "$DIM" "$OFF"
    printf '%s      run ./scripts/download_gpt2.sh for full coverage%s\n' "$DIM" "$OFF"
  fi
  # A superset of the default suite: `council` adds tests/council.rs and
  # `train` adds autograd/training/train_ops, all of which cargo would
  # otherwise skip on required-features.
  run "cargo test --release"       cargo test --release --locked --features council,train
fi

printf '\n%s%s✓ all %s checks passed%s %s(%ss)%s\n' \
  "$GREEN" "$BOLD" "$STAGE" "$OFF" "$DIM" "$((SECONDS - started))" "$OFF"
