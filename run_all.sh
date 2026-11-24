#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

mode="${1:-all}"
if (( $# )); then shift; fi
case "$mode" in
  all|baseline|bert) ;;
  *) echo "Cách dùng: bash run_all.sh [all|baseline|bert] [cờ chung]" >&2; exit 2 ;;
esac

python_bin="${PYTHON_BIN:-python}"
data_dir="${DATA_DIR:-data/processed}"
output_dir="${OUTPUT_DIR:-output/sentiment}"
common=(--data-dir "$data_dir" --output-dir "$output_dir")
while (( $# )); do
  case "$1" in
    --output-dir)
      if (( $# < 2 )); then echo "Thiếu giá trị cho --output-dir" >&2; exit 2; fi
      output_dir="$2"
      common+=("$1" "$2")
      shift 2
      ;;
    --output-dir=*) output_dir="${1#*=}"; common+=("$1"); shift ;;
    *) common+=("$1"); shift ;;
  esac
done
read -r -a models <<< "${BERT_MODELS:-uitnlp/visobert vinai/phobert-base-v2 xlm-roberta-base}"
read -r -a rates <<< "${BERT_LEARNING_RATES:-2e-5}"
read -r -a seeds <<< "${BERT_SEEDS:-42 2025 3407}"

variants=(main)
if [[ "${ABLATIONS:-1}" == 1 ]]; then
  variants+=(no_emote_emoji mask_entities no_class_weight)
fi

for variant in "${variants[@]}"; do
  settings=(--variant "$variant" --emoji-mode keep --no-mask-entities --class-weight)
  case "$variant" in
    no_emote_emoji) settings+=(--emoji-mode remove) ;;
    mask_entities) settings+=(--mask-entities) ;;
    no_class_weight) settings+=(--no-class-weight) ;;
  esac
  if [[ "$mode" != bert ]]; then
    "$python_bin" train_tfidf_lr.py "${common[@]}" "${settings[@]}" \
      --seed "${LR_SEED:-42}" --search "${TFIDF_SEARCH:-random}" \
      --search-iterations "${TFIDF_SEARCH_ITERATIONS:-12}" --n-jobs "${TFIDF_JOBS:-1}"
  fi
  if [[ "$mode" != baseline ]]; then
    for model in "${models[@]}"; do
      "$python_bin" train_bert.py "${common[@]}" "${settings[@]}" \
        --model-name "$model" --seeds "${seeds[@]}" --learning-rates "${rates[@]}" \
        --batch-size "${BERT_BATCH_SIZE:-32}" --eval-batch-size "${BERT_EVAL_BATCH_SIZE:-32}" \
        --gradient-accumulation-steps "${BERT_GRAD_ACCUM:-1}" --epochs "${BERT_EPOCHS:-10}"
    done
  fi
done

"$python_bin" evaluate.py --output-dir "$output_dir"
