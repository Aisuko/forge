#!/usr/bin/env bash
# The single definition of what "green" means here. The hooks in .githooks/
# and .github/workflows/ci.yml are thin wrappers around it, so nothing can
# drift from it.
#
#   ./scripts/local/ci_local.sh fast   # fmt, clippy, builds, dep assert, the site
#   ./scripts/local/ci_local.sh full   # everything in fast, plus the release tests
#
# `fast` runs on pre-commit (~7s warm), `full` on pre-push (~1m10s). Every
# optional feature and every tool gets its own run: one nobody builds rots.
#
# Cheapest first, stopping at the first failure, so a formatting slip does not
# cost a minute of GPU tests.
set -uo pipefail
cd "$(dirname "$0")/../.."

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

# The TUI deps live in tools/forge-top's manifest so they can never reach a
# dependent of the library, or the wasm build.
assert_no_tui_deps() {
  local tree dep n
  tree=$(cargo tree -e normal --locked -p forge-ml) || return 1
  for dep in ratatui crossterm sysinfo nvml-wrapper memmap2; do
    n=$(grep -c "^.*[^a-z-]$dep v" <<<"$tree" || true)
    if [[ "$n" -ne 0 ]]; then
      echo "$dep reached forge-ml's dependency tree" >&2
      return 1
    fi
  done
}

printf '%sforge local CI — %s stage%s\n\n' "$BOLD" "$STAGE" "$OFF"

run "cargo fmt --check"            cargo fmt --all --check
run "cargo clippy -D warnings"     cargo clippy -p forge-ml --all-targets --locked -- -D warnings
# `train` is off by default, so autograd, optim, the nine backward kernels and
# four targets are invisible to the pass above. Lint them explicitly.
run "clippy --features train"      cargo clippy -p forge-ml --all-targets --locked --features train -- -D warnings
run "clippy tools"                 cargo clippy -p forge-council -p forge-top --all-targets --locked -- -D warnings
run "build wasm32"                 cargo build --release --locked --target wasm32-unknown-unknown -p forge-ml
run "build wasm32 (council)"       cargo build --release --locked --target wasm32-unknown-unknown -p forge-council
run "build forge-top"              cargo build --release --locked -p forge-top
run "no TUI deps in forge-ml"      assert_no_tui_deps
# ~0.5s: both wasm crates are already built above, so this is wasm-bindgen,
# Tailwind and the copies. It catches a page naming an asset that moved, which
# nothing else here would see. Whether the pages *run* is `make site-verify`.
run "build site"                   ./scripts/common/build_site.sh

if [[ "$STAGE" == full ]]; then
  # gpt2_e2e, kv_cache and the BPE cases self-skip without models/gpt2/, which
  # makes a green run quietly weaker than it looks. Say so.
  if [[ ! -f models/gpt2/model.safetensors ]]; then
    printf '%snote: models/gpt2/ missing — gpt2_e2e and kv_cache will self-skip%s\n' "$DIM" "$OFF"
    printf '%s      run ./scripts/local/download_gpt2.sh for full coverage%s\n' "$DIM" "$OFF"
  fi
  # `train` adds autograd/training/train_ops; cargo skips them without it.
  run "cargo test --release"       cargo test -p forge-ml --release --locked --features train
  run "cargo test (tools)"         cargo test -p forge-council --release --locked
fi

printf '\n%s%s✓ all %s checks passed%s %s(%ss)%s\n' \
  "$GREEN" "$BOLD" "$STAGE" "$OFF" "$DIM" "$((SECONDS - started))" "$OFF"
