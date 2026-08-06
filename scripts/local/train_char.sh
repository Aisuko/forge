#!/usr/bin/env bash
# Train the character-level Shakespeare model, evaluate it, and tell you
# whether it is worth shipping — in one command.
#
#   ./scripts/local/train_char.sh                 # the full campaign (~10 h on an A5000)
#   ./scripts/local/train_char.sh --baseline-only # score what already exists (~4 min)
#   ./scripts/local/train_char.sh --quick         # 600-step smoke test of the whole flow
#   ./scripts/local/train_char.sh --report        # re-render the report, no training
#
# What it does, in order:
#   0. scores every checkpoint you already have, including the one the website
#      currently serves, so there is a number to beat
#   1. trains several recipes, each keeping its own *best* checkpoint rather
#      than its last, and stopping when validation loss stops improving
#   2. scores every result on held-out loss, spelling, and a memorisation guard
#   3. names a champion and prints the command that ships it
#
# Safe to interrupt: re-running skips any run that already finished.
#
# Why "best rather than last" is the whole point: on this recipe validation
# loss bottoms out near step 2000 of 5000 and then climbs while training loss
# keeps falling. The final checkpoint is the worst one the run produced. Every
# model in checkpoints/ before this script existed was a final checkpoint, and
# the one the site serves scores 1.73 against the 1.47 of a checkpoint that was
# sitting unshipped next to it.
set -uo pipefail
cd "$(dirname "$0")/../.."

# ── knobs ────────────────────────────────────────────────────────────────────

OUT="${OUT:-checkpoints/campaign}"
BACKEND="${BACKEND:-wgpu}"
STEPS="${STEPS:-5000}"
EVAL_EVERY="${EVAL_EVERY:-100}"
# Evaluations without improvement before a run gives up. 12 × 100 steps is
# ~54 minutes of patience on an A5000.
PATIENCE="${PATIENCE:-12}"
EVAL_WINDOWS="${EVAL_WINDOWS:-512}"

# Sampling used for every quality measurement. Fixed on purpose: two models
# compared under different sampling are not being compared.
TOPK=12
TEMP=0.8
GEN_TOKENS=250
SEEDS=(7 11 13)
PROMPTS=("ROMEO:" "JULIET:" "KING RICHARD III:" "First Citizen:" "HAMLET:"
         "MENENIUS:" "CORIOLANUS:" "QUEEN MARGARET:" "GLOUCESTER:" "BRUTUS:"
         "Second Servingman:" "DUKE VINCENTIO:")

# Composite score, in nats — lower is better. See score_model() for the shape.
W_WORD=4.0
WORD_TARGET=0.98
# A candidate reproducing the corpus is rejected outright, however well it
# scores on everything else. Not currently binding: measured copy32 is 0.000
# and the longest shared span 24-27 characters across every model tested.
MAX_COPY32=0.02
MAX_COPY_LEN=96

# The search. Step count is not the lever — validation loss bottoms early
# whatever you set it to — so these vary the regularisation, which moves where
# the bottom is. Override with RUNS="name:flags" pairs.
DEFAULT_RUNS=(
  "base:--dropout 0.2 --wd 0.1 --lr 1e-3 --seed 1337"
  "drop30:--dropout 0.3 --wd 0.1 --lr 1e-3 --seed 1337"
  "drop40:--dropout 0.4 --wd 0.1 --lr 1e-3 --seed 1337"
  "drop30wd20:--dropout 0.3 --wd 0.2 --lr 1e-3 --seed 1337"
  "drop30lr6e4:--dropout 0.3 --wd 0.1 --lr 6e-4 --seed 1337"
)

MODE=full
case "${1:-}" in
  --baseline-only) MODE=baseline ;;
  --report)        MODE=report ;;
  --quick)         MODE=full; STEPS=600; PATIENCE=3; EVAL_EVERY=50
                   DEFAULT_RUNS=("base:--dropout 0.2 --wd 0.1 --lr 1e-3 --seed 1337"
                                 "drop30:--dropout 0.3 --wd 0.1 --lr 1e-3 --seed 1337") ;;
  "")              ;;
  *) echo "usage: $0 [--baseline-only|--quick|--report]" >&2; exit 2 ;;
esac

IFS=$'\n' read -rd '' -a RUNS <<<"${RUNS:-$(printf '%s\n' "${DEFAULT_RUNS[@]}")}" || true

TRAIN=target/release/examples/train_shakespeare
GEN=target/release/examples/generate
SCORES="$OUT/scores.tsv"
REPORT="$OUT/report.md"

if [[ -t 1 ]]; then B=$'\033[1m'; G=$'\033[32m'; D=$'\033[2m'; O=$'\033[0m'
else B=''; G=''; D=''; O=''; fi

say() { printf '%s%s%s\n' "$B" "$*" "$O"; }

# ── setup ────────────────────────────────────────────────────────────────────

[[ -f data/tinyshakespeare.txt ]] || {
  echo "data/tinyshakespeare.txt missing — run ./scripts/local/download_shakespeare.sh" >&2
  exit 1
}
mkdir -p "$OUT"

if [[ $MODE != report ]]; then
  say "== building"
  cargo build --release --features train --example train_shakespeare --example generate || exit 1
fi

# ── scoring ──────────────────────────────────────────────────────────────────

# score_model <label> <model-dir-or-checkpoint>
#
# Appends "label<TAB>val<TAB>train<TAB>word<TAB>copy32<TAB>maxcopy<TAB>score<TAB>verdict"
# to $SCORES. Accepts either a directory holding model.fzm or model.safetensors
# (what assets/ and the ship script use — assets/ ships q4 now, so this is the
# score of the quantized file the site serves) or a bare .safetensors path with
# its sidecars beside it (what training writes).
score_model() {
  local label="$1" path="$2" dir ckpt tmp
  tmp="$(mktemp -d)"
  if [[ -d "$path" ]]; then
    dir="$path"
    local src=""
    for cand in "$dir/model.fzm" "$dir/model.safetensors"; do
      [[ -f "$cand" ]] && { src="$cand"; break; }
    done
    [[ -n "$src" ]] || { echo "  skip $label: no model.fzm or model.safetensors" >&2
                         rm -rf "$tmp"; return 1; }
    ckpt="$tmp/probe.${src##*.}"
    cp "$src" "$ckpt"
    cp "$dir/config.json" "$tmp/probe.config.json"
    cp "$dir/vocab.json" "$tmp/probe.vocab.json"
  else
    ckpt="$path"
    local stem; stem="$(basename "$path" .safetensors)"
    local base; base="$(dirname "$path")"
    [[ -f "$base/$stem.config.json" ]] || { echo "  skip $label: no $stem.config.json" >&2
                                            rm -rf "$tmp"; return 1; }
    # generate wants a directory laid out the way the site's model/ is.
    dir="$tmp/model"; mkdir -p "$dir"
    cp "$path" "$dir/model.safetensors"
    cp "$base/$stem.config.json" "$dir/config.json"
    cp "$base/$stem.vocab.json" "$dir/vocab.json"
  fi

  # A char model only: a BPE checkpoint would load against the wrong vocab and
  # return a meaningless number rather than an error.
  if ! grep -q '"tokenizer": *"char"' "$dir/config.json" 2>/dev/null; then
    echo "  skip $label: not a char-vocab checkpoint" >&2; rm -rf "$tmp"; return 1
  fi

  # Architecture from the checkpoint's own sidecar, not from the trainer's
  # defaults: --eval-only builds its config from the CLI, so a model trained at
  # a different width would otherwise fail to load with a shape error rather
  # than being evaluated.
  local nl nh ne
  nl=$(sed -n 's/.*"n_layer": *\([0-9]*\).*/\1/p' "$dir/config.json")
  nh=$(sed -n 's/.*"n_head": *\([0-9]*\).*/\1/p' "$dir/config.json")
  ne=$(sed -n 's/.*"n_embd": *\([0-9]*\).*/\1/p' "$dir/config.json")

  local losses val train
  losses=$("$TRAIN" --backend "$BACKEND" --eval-only --eval-windows "$EVAL_WINDOWS" \
             --layers "${nl:-6}" --heads "${nh:-6}" --embd "${ne:-384}" \
             --checkpoint "$ckpt" 2>/dev/null | grep '^{')
  [[ -n "$losses" ]] || { echo "  skip $label: eval failed" >&2; rm -rf "$tmp"; return 1; }
  val=$(sed 's/.*"val":\([0-9.]*\).*/\1/' <<<"$losses")
  train=$(sed 's/.*"train":\([0-9.]*\).*/\1/' <<<"$losses")

  # The identical sample set for every candidate, so a difference is the model
  # and not the draw.
  : > "$tmp/samples.txt"
  for seed in "${SEEDS[@]}"; do
    for p in "${PROMPTS[@]}"; do
      "$GEN" --model "$dir" --backend "$BACKEND" --prompt "$p" --tokens "$GEN_TOKENS" \
             --topk "$TOPK" --temp "$TEMP" --seed "$seed" 2>/dev/null \
        | sed -n '/^--- output/,$p' | tail -n +2 >> "$tmp/samples.txt"
    done
  done

  local metrics
  metrics=$(python3 scripts/local/score_text.py data/tinyshakespeare.txt "$tmp/samples.txt") || {
    echo "  skip $label: scoring failed" >&2; rm -rf "$tmp"; return 1; }
  rm -rf "$tmp"

  # word<TAB>copy32<TAB>maxcopy<TAB>speaker
  local word copy32 maxcopy
  word=$(cut -f1 <<<"$metrics"); copy32=$(cut -f2 <<<"$metrics"); maxcopy=$(cut -f3 <<<"$metrics")

  local score verdict
  score=$(python3 -c "
import sys
val,word=float('$val'),float('$word')
print(f'{val + $W_WORD * max(0.0, $WORD_TARGET - word):.4f}')")
  verdict=ok
  awk "BEGIN{exit !($copy32 > $MAX_COPY32)}" && verdict="REJECTED: reproduces the corpus"
  (( maxcopy > MAX_COPY_LEN )) && verdict="REJECTED: ${maxcopy}-char verbatim span"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$val" "$train" "$word" "$copy32" "$maxcopy" "$score" "$verdict" >> "$SCORES"
  printf '  %-22s val %-7s word %-6s score %s %s%s%s\n' \
    "$label" "$val" "$word" "$score" "$D" "$verdict" "$O"
}

# ── phase 0: what already exists ─────────────────────────────────────────────

if [[ $MODE != report ]]; then
  : > "$SCORES"
  say "== phase 0: scoring what you already have"
  [[ -d assets/shakespeare_char ]] && score_model "LIVE (assets/)" assets/shakespeare_char
  for f in checkpoints/*.safetensors; do
    [[ -e "$f" ]] || continue
    score_model "existing: $(basename "$f" .safetensors)" "$f"
  done
fi

# ── phase 1 + 2: train, then score ───────────────────────────────────────────

if [[ $MODE == full ]]; then
  say "== phase 1: training ${#RUNS[@]} recipes (${STEPS} steps each, early stop after ${PATIENCE} flat evals)"
  echo "${D}each run keeps its own best checkpoint, not its last${O}"
  for spec in "${RUNS[@]}"; do
    name="${spec%%:*}"; flags="${spec#*:}"
    rundir="$OUT/$name"; mkdir -p "$rundir"
    ckpt="$rundir/m.safetensors"
    if [[ -f "$rundir/DONE" ]]; then
      echo "  ${D}$name already finished — skipping${O}"; continue
    fi
    say "-- $name  ($flags)"
    # shellcheck disable=SC2086
    if "$TRAIN" --backend "$BACKEND" --tokenizer char --steps "$STEPS" \
        --eval-every "$EVAL_EVERY" --eval-windows "$EVAL_WINDOWS" \
        --early-stop "$PATIENCE" --sample-every 0 \
        --checkpoint "$ckpt" $flags 2>&1 | tee "$rundir/train.log"; then
      touch "$rundir/DONE"
    else
      echo "  $name failed — see $rundir/train.log" >&2
    fi
  done

  say "== phase 2: scoring each run's best checkpoint"
  for spec in "${RUNS[@]}"; do
    name="${spec%%:*}"
    best="$OUT/$name/m.best.safetensors"
    [[ -f "$best" ]] && score_model "run: $name" "$best"
  done
fi

# ── phase 3: the report ──────────────────────────────────────────────────────

[[ -s "$SCORES" ]] || { echo "nothing scored — see the errors above" >&2; exit 1; }

python3 - "$SCORES" "$REPORT" "$OUT" <<'PY'
import sys, pathlib, datetime

scores, report, out = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
rows = []
for line in pathlib.Path(scores).read_text().splitlines():
    if not line.strip():
        continue
    label, val, train, word, copy32, maxcopy, score, verdict = line.split("\t")
    rows.append(dict(label=label, val=float(val), train=float(train),
                     word=float(word), copy32=float(copy32), maxcopy=int(maxcopy),
                     score=float(score), verdict=verdict))

live = next((r for r in rows if r["label"].startswith("LIVE")), None)
ok = [r for r in rows if r["verdict"] == "ok"]
ok.sort(key=lambda r: r["score"])
champ = ok[0] if ok else None

lines = [
    "# char-model campaign report",
    "",
    f"Generated {datetime.date.today().isoformat()}.",
    "",
    "`score = val_loss + 4.0 x max(0, 0.98 - word_validity)`, lower is better —",
    "held-out loss with a penalty for broken spelling, both in nats. A candidate",
    "reproducing the training corpus is rejected outright regardless of score.",
    "",
    "| model | val | train | word ok | copy32 | longest copy | **score** | |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
]
for r in sorted(rows, key=lambda r: r["score"]):
    mark = " **← champion**" if champ and r is champ else ""
    note = "" if r["verdict"] == "ok" else f" {r['verdict']}"
    lines.append(
        f"| {r['label']} | {r['val']:.4f} | {r['train']:.4f} | {r['word']:.3f} | "
        f"{r['copy32']:.3f} | {r['maxcopy']} | **{r['score']:.4f}** |{mark}{note} |"
    )

lines += ["", "nanoGPT's published reference for this configuration is val 1.4697.", ""]

if champ is None:
    lines += ["## No shippable candidate", "",
              "Every candidate was rejected by the memorisation guard."]
elif live and champ["label"].startswith("LIVE"):
    lines += ["## Nothing beat what is already live", "",
              f"The live model still scores best at {champ['score']:.4f}. Ship nothing."]
else:
    delta = (live["score"] - champ["score"]) if live else float("nan")
    src = champ["label"].split(": ", 1)[-1]
    path = (f"{out}/{src}/m.best.safetensors" if champ["label"].startswith("run:")
            else f"checkpoints/{src}.safetensors")
    lines += [
        f"## Champion: {champ['label']}",
        "",
        f"- score **{champ['score']:.4f}**"
        + (f", an improvement of **{delta:.4f}** over the live model's {live['score']:.4f}"
           if live else ""),
        f"- held-out validation loss **{champ['val']:.4f}**, train {champ['train']:.4f}",
        f"- {champ['word']:.1%} of generated words appear in the corpus vocabulary",
        f"- longest span shared with the training text: {champ['maxcopy']} characters",
        "",
        "Ship it:",
        "",
        "```bash",
        f"./scripts/local/ship_char_model.sh {path}",
        "cargo run --release --example gate_tokens -- --model assets/shakespeare_char",
        "```",
        "",
        "The second command matters: `tests/data/gate_expected.json` holds greedy",
        "tokens from the *current* weights, and is the reference a browser run is",
        "compared against. New weights without a new fixture is a stale fixture.",
    ]

report.write_text("\n".join(lines) + "\n")
# Everything from the table header to the end of the verdict — the whole point
# of running this is the ranking and the ship command, so print both.
print("\n".join(lines[7:]))
print(f"\nfull report: {report}")
PY
