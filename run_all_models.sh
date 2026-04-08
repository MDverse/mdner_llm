set -euo pipefail

MODELS=(
  "openai/gpt-5.4"
  "openai/gpt-4o"
  "openai/gpt-oss-120b"
  "deepseek/deepseek-v3.2"
  "nvidia/nemotron-3-super-120b-a12b"
  "qwen/qwen3.5-122b-a10b"
  "google/gemini-3.1-pro-preview"
  "mistralai/mistral-large-2512"
  "meta-llama/llama-4-maverick"
  "anthropic/claude-sonnet-4.6"
)

FRAMEWORKS=(
  #"none"
  "instructor"
  #"pydanticai"
)

PROMPT_FILE="json_few_shot.txt"
TEXT_PATH="results/groundtruth_paths.txt"
OUTPUT_DIR="results/llm/annotations"
TAG_PROMPT="json"
MAX_RETRIES=3

for model in "${MODELS[@]}"; do
  for framework in "${FRAMEWORKS[@]}"; do
    echo "Running model= ${model} framework=${framework}"

    uv run src/mdner_llm/core/extract_entities_all_texts.py \
      --prompt-file "${PROMPT_FILE}" \
      --model "${model}" \
      --texts-path "${TEXT_PATH}" \
      --prompt-tag "${TAG_PROMPT}" \
      --framework "${framework}" \
      --output-dir "${OUTPUT_DIR}" \
      --max-retries "${MAX_RETRIES}"
  done
done
