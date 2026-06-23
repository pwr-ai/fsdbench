#!/usr/bin/env bash
# Run the benchmark on the HARD split (data/raw-hard.json, 250 samples).
# Launches 25 parallel processes of 10 samples each (0-249).
#
# Usage:  bash scripts/run_hard.sh [PROMPT]
#   PROMPT defaults to "baseline" (alternatives: "main").
set -euo pipefail

SPLIT="hard"
DATASET="data/raw-${SPLIT}.json"
LOG_DIR="logs-hard"
MODEL="${CHATBOT_MODEL:-anthropic/claude-sonnet-4-6}"
MAX_ROUNDS="${MAX_ROUNDS:-50}"
BATCH="${BATCH:-10}"
PROMPT="${1:-baseline}"

mkdir -p "$LOG_DIR" data

# Fetch the split from the Hugging Face Hub if not already present locally
# (requires `pip install datasets`).
if [ ! -f "$DATASET" ]; then
    echo "Fetching '$SPLIT' split from AI-TAX/factual-state-discovery-benchmark -> $DATASET"
    python -c "import json; from datasets import load_dataset; json.dump(load_dataset('AI-TAX/factual-state-discovery-benchmark', split='${SPLIT}').to_list(), open('${DATASET}','w'), ensure_ascii=False)"
fi

for i in $(seq 0 24); do
    START=$((i * BATCH))
    fsdbench run \
        --start_sample "$START" --num_samples "$BATCH" \
        --chatbot_model "$MODEL" --max_rounds "$MAX_ROUNDS" \
        --no_gaps_hint --prompt "$PROMPT" \
        --dataset "$DATASET" --log_dir "$LOG_DIR" \
        > "${LOG_DIR}/${PROMPT}_s${START}_n${BATCH}.log" 2>&1 &
done

wait
echo "Launched 25 processes (prompt=$PROMPT)"
echo "Logs in: $LOG_DIR"
