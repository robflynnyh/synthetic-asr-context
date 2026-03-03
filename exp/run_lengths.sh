#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

python3.10 "${SCRIPT_DIR}/avg_utterance_words.py" \
  --dataset tedlium3 \
  --split test \
  --output_file "${SCRIPT_DIR}/tedlium.length.txt"

python3.10 "${SCRIPT_DIR}/avg_utterance_words.py" \
  --dataset gigaspeech \
  --split test \
  --output_file "${SCRIPT_DIR}/gigaspeech.length.txt"

python3.10 "${SCRIPT_DIR}/avg_utterance_words.py" \
  --dataset librispeech \
  --split test-other \
  --output_file "${SCRIPT_DIR}/librispeech.length.txt"

echo "Saved outputs to:"
echo "  ${SCRIPT_DIR}/tedlium.length.txt"
echo "  ${SCRIPT_DIR}/gigaspeech.length.txt"
echo "  ${SCRIPT_DIR}/librispeech.length.txt"
