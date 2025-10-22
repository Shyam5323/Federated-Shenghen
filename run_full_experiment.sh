#!/bin/bash
set -euo pipefail

STRATEGY=${1:-iid}
LOG_DIR="logs"
PARTITION_LOG="$LOG_DIR/partition.log"
CENTRAL_LOG="$LOG_DIR/central.log"
EVAL_LOG="$LOG_DIR/evaluation_${STRATEGY}.log"

mkdir -p "$LOG_DIR"

: > "$PARTITION_LOG"
: > "$CENTRAL_LOG"
: > "$EVAL_LOG"

printf "==============================\n"
printf "Shenzhen Federated Experiment\n"
printf "==============================\n"
printf "Strategy: %s\n\n" "$STRATEGY"

printf "[1/4] Generating partitions...\n"
python partition.py | tee "$PARTITION_LOG"

printf "[2/4] Training centralized model...\n"
python central_mura.py | tee "$CENTRAL_LOG"

printf "[3/4] Running federated experiment (%s)...\n" "$STRATEGY"
./run_federated.sh "$STRATEGY"

printf "[4/4] Evaluating models on held-out test set...\n"
python evaluate_models.py --strategy "$STRATEGY" | tee "$EVAL_LOG"

printf "\nDone. Review logs in %s and metrics in the analysis/ directory.\n" "$LOG_DIR"
