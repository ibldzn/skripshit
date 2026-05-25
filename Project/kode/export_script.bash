#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="./exported-onnx"
IMGSZ=224
WEIGHT_FILE="best.pt"

mkdir -p "$OUT_DIR"

export_one() {
  local model_label="$1"
  local run_dir="$2"
  local seed="$3"

  local pt_path="${run_dir}/seed-${seed}/weights/${WEIGHT_FILE}"
  local generated_onnx="${pt_path%.pt}.onnx"
  local target_onnx="${OUT_DIR}/${model_label}-seed-${seed}.onnx"

  echo "Exporting ${model_label} seed ${seed}"
  echo "PT     : ${pt_path}"
  echo "Target : ${target_onnx}"

  if [[ ! -f "$pt_path" ]]; then
    echo "ERROR: PT file not found: $pt_path"
    exit 1
  fi

  yolo export model="$pt_path" format=onnx imgsz="$IMGSZ"

  if [[ ! -f "$generated_onnx" ]]; then
    echo "ERROR: ONNX file not found: $generated_onnx"
    exit 1
  fi

  cp "$generated_onnx" "$target_onnx"
}

for seed in {1..20}; do
  export_one "yolov8n" "runs/classify/runs/yolov8n_seed_experiment" "$seed"
  export_one "yolo11n" "runs/classify/runs/yolo11n_seed_experiment" "$seed"
done

echo "Done. Output: $OUT_DIR"
