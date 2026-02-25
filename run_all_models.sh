#!/usr/bin/env bash
set -euo pipefail

MODELS=(
  "openai/gpt-5.1"
  "openai/gpt-4o"
  "openai/gpt-oss-120b"
  "meta-llama/llama-4-maverick"
  "google/gemini-3-pro-preview"
  "mistralai/mistral-large-2512"
  "qwen/qwen3-vl-30b-a3b-thinking"
  "deepseek/deepseek-v3.2"
)

FRAMEWORKS=(
  "none"
  "instructor"
  "llamaindex"
  "pydanticai"
)

PROMPT_PATH="prompts/json_few_shot.txt"
TEXT_PATH="results/2_selected_files_20260106_142515.txt"
OUTPUT_DIR="results/llm_annotations"
TAG_PROMPT="json"
MAX_RETRIES=3

for model in "${MODELS[@]}"; do
  for framework in "${FRAMEWORKS[@]}"; do
    echo "Running model=${model} framework=${framework}"

    uv run src/extract_entities_all_texts.py \
      --path-prompt "${PROMPT_PATH}" \
      --model "${model}" \
      --path-texts "${TEXT_PATH}" \
      --tag-prompt "${TAG_PROMPT}" \
      --framework "${framework}" \
      --output-dir "${OUTPUT_DIR}" \
      --max-retries "${MAX_RETRIES}"
  done
done
