#!/bin/bash
# GPU_ID=1 WHISPER_MODEL=openai/whisper-large-v3-turbo LOG_PREFIX=turbo_caps bash run_evals.sh
# CAPS-history eval sweep
# Runs context lengths 1..23 for each dataset using --uppercase_history.

PYTHON_BIN=${PYTHON_BIN:-python3.10}
GPU_ID=${GPU_ID:-2}
OUT_DIR=${OUT_DIR:-./results_history}
WHISPER_MODEL=${WHISPER_MODEL:-openai/whisper-tiny.en}
LOG_PREFIX=${LOG_PREFIX:-}
START_UTTERANCES=${START_UTTERANCES:-1}
END_UTTERANCES=${END_UTTERANCES:-23}

if [ -n "$LOG_PREFIX" ]; then
  LOG_PREFIX="${LOG_PREFIX}_"
fi

if [ "$START_UTTERANCES" -gt "$END_UTTERANCES" ]; then
  echo "START_UTTERANCES must be <= END_UTTERANCES"
  exit 1
fi

mkdir -p "$OUT_DIR"

for k in $(seq "$START_UTTERANCES" "$END_UTTERANCES"); do
  echo "${k}/${END_UTTERANCES} on tedlium3"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" eval_hf.py \
    --use_history --sequential \
    --dataset tedlium3 --split test \
    --whisper_model "$WHISPER_MODEL" \
    --prev_utterances "$k" \
    --uppercase_history \
    --log_path "$OUT_DIR/${LOG_PREFIX}ted_history_caps_${k}.txt"
done
echo "Finished tedlium3"

for k in $(seq "$START_UTTERANCES" "$END_UTTERANCES"); do
  echo "${k}/${END_UTTERANCES} on librispeech"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" eval_hf.py \
    --use_history --sequential \
    --dataset librispeech --split test-other \
    --whisper_model "$WHISPER_MODEL" \
    --prev_utterances "$k" \
    --uppercase_history \
    --log_path "$OUT_DIR/${LOG_PREFIX}LS_history_caps_${k}.txt"
done
echo "Finished librispeech"

for k in $(seq "$START_UTTERANCES" "$END_UTTERANCES"); do
  echo "${k}/${END_UTTERANCES} on gigaspeech"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" eval_hf.py \
    --use_history --sequential \
    --dataset gigaspeech --split test \
    --whisper_model "$WHISPER_MODEL" \
    --prev_utterances "$k" \
    --reset_history_on_pause \
    --uppercase_history \
    --log_path "$OUT_DIR/${LOG_PREFIX}GS_history_rsp_caps_${k}.txt"
done
echo "Finished gigaspeech"
