#!/usr/bin/env bash
#
# Sequential A/B training runs for data/fssp-digits5.
#
# Defaults are chosen to compare architectures on the same dataset, input size,
# preprocessing, augmentation and dropout. Each run writes its own artifacts and
# log file, then the script prints a compact summary of best validation scores.
#
# Usage:
#   ./fssp-train.sh
#
# Useful overrides:
#   BIN=./target/debug/capburn ./fssp-train.sh
#   BACKEND=cpu EPOCHS=10 ./fssp-train.sh
#   INCLUDE_FIT=1 ./fssp-train.sh       # also compare fit preprocessing
#   DROPOUTS="0.3 0.4" ./fssp-train.sh
set -euo pipefail
cd "$(dirname "$0")"

BIN="${BIN:-./target/debug/capburn}"
BACKEND="${BACKEND:-wgpu}"
DATA_ROOT="${DATA_ROOT:-./data}"
DATASET="${DATASET:-fssp-digits5}"
OUT_ROOT="${OUT_ROOT:-./artifacts/fssp-digits5-runs}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-0.0005}"
AUGMENT="${AUGMENT:-medium}"
INPUT_SIZE="${INPUT_SIZE:-auto}"
DROPOUTS="${DROPOUTS:-0.3}"
INCLUDE_FIT="${INCLUDE_FIT:-0}"

mkdir -p "$OUT_ROOT/logs"

if [ ! -x "$BIN" ]; then
    echo ">>> $BIN not found; building debug trainer"
    cargo build -p capburn-train
fi

ARCHES=(fixed fixed-global fixed-seq fixed-seq-pool)

PREPROCESSES=(stretch)
if [ "$INCLUDE_FIT" = "1" ]; then
    PREPROCESSES+=(fit)
fi

echo ">>> Dataset:     $DATA_ROOT/$DATASET"
echo ">>> Backend:     $BACKEND"
echo ">>> Binary:      $BIN"
echo ">>> Output root: $OUT_ROOT"
echo ">>> Input size:  $INPUT_SIZE"
echo ">>> Epochs:      $EPOCHS"
echo ">>> Batch size:  $BATCH_SIZE"
echo ">>> Augment:     $AUGMENT"
echo ">>> Dropouts:    $DROPOUTS"
echo

run_one() {
    local arch="$1"
    local preprocess="$2"
    local dropout="$3"
    local drop_tag="${dropout/./}"
    local name="${DATASET}-${arch}-${INPUT_SIZE}-${preprocess}-drop${drop_tag}"
    local out="$OUT_ROOT/$name"
    local log="$OUT_ROOT/logs/$name.log"

    echo ">>> START $name"
    echo ">>> log: $log"
    "$BIN" train \
        --backend "$BACKEND" \
        --data "$DATA_ROOT" \
        --dataset "$DATASET" \
        --arch "$arch" \
        --preprocess "$preprocess" \
        --input-size "$INPUT_SIZE" \
        --augment "$AUGMENT" \
        --dropout "$dropout" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --out "$out" 2>&1 | tee "$log"
    echo ">>> DONE $name"
    echo
}

for dropout in $DROPOUTS; do
    for preprocess in "${PREPROCESSES[@]}"; do
        for arch in "${ARCHES[@]}"; do
            run_one "$arch" "$preprocess" "$dropout"
        done
    done
done

echo ">>> Summary"
grep -h "Done. Best validation accuracy" "$OUT_ROOT"/logs/*.log \
    | sed -E 's/^/  /' || true
echo ">>> Logs: $OUT_ROOT/logs"
