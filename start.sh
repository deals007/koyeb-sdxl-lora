#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# Settings you can change
# -----------------------
MODEL_URL="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true"
MODEL_PATH="/workspace/models/sd_xl_base_1.0.safetensors"

DATA_ZIP="/workspace/bimbo-nails.zip"
DATA_RAW="/workspace/dataset_raw"
TRAIN_DIR="/workspace/dataset/train/20_bimbonails"

OUT_DIR="/workspace/output"
OUT_NAME="bimbonails_sdxl_lora"

# Training params (safe defaults)
STEPS="${STEPS:-3000}"
RANK="${RANK:-16}"
ALPHA="${ALPHA:-16}"

mkdir -p /workspace/models /workspace/dataset /workspace/output

# -----------------------
# 1) Download SDXL base model (with basic size sanity check)
# -----------------------
if [ ! -f "$MODEL_PATH" ]; then
  echo "Downloading SDXL base model..."
  curl -fL --retry 10 --retry-delay 5 -o "${MODEL_PATH}.partial" "$MODEL_URL"
  BYTES=$(stat -c%s "${MODEL_PATH}.partial" || echo 0)
  echo "Downloaded bytes: $BYTES"
  # SDXL base is ~6.5GB. If it's tiny, it likely downloaded HTML/blocked.
  if [ "$BYTES" -lt 1000000000 ]; then
    echo "ERROR: Model download too small. First 300 bytes:"
    head -c 300 "${MODEL_PATH}.partial" || true
    exit 1
  fi
  mv "${MODEL_PATH}.partial" "$MODEL_PATH"
else
  echo "Model exists: $MODEL_PATH"
fi

# -----------------------
# 2) Unzip dataset
# -----------------------
rm -rf "$DATA_RAW"
mkdir -p "$DATA_RAW"
unzip -o "$DATA_ZIP" -d "$DATA_RAW"

# Your zip extracts to: /workspace/dataset_raw/longnails-lora
SRC_DIR="$(find "$DATA_RAW" -maxdepth 2 -type d -name "longnails-lora" | head -n 1)"
if [ -z "$SRC_DIR" ]; then
  echo "ERROR: Could not find extracted folder 'longnails-lora' inside zip."
  echo "Folders found:"
  find "$DATA_RAW" -maxdepth 2 -type d
  exit 1
fi

# -----------------------
# 3) Fix trigger token (recommended)
#    Change 'bimbo nails' -> 'bimbonails'
# -----------------------
echo "Updating captions trigger: 'bimbo nails' -> 'bimbonails' ..."
find "$SRC_DIR" -type f -name "*.txt" -print0 | while IFS= read -r -d '' f; do
  sed -i 's/\bbimbo nails\b/bimbonails/g' "$f"
done

# -----------------------
# 4) Move into kohya repeats folder structure
# -----------------------
rm -rf "$TRAIN_DIR"
mkdir -p "$TRAIN_DIR"
# move images + txt
shopt -s nullglob
mv "$SRC_DIR"/* "$TRAIN_DIR"/

# Basic sanity check: counts match
PNG_COUNT=$(ls -1 "$TRAIN_DIR"/*.png 2>/dev/null | wc -l || true)
TXT_COUNT=$(ls -1 "$TRAIN_DIR"/*.txt 2>/dev/null | wc -l || true)
echo "Dataset: PNG=$PNG_COUNT TXT=$TXT_COUNT"
if [ "$PNG_COUNT" -eq 0 ] || [ "$PNG_COUNT" -ne "$TXT_COUNT" ]; then
  echo "ERROR: PNG/TXT mismatch or empty dataset."
  exit 1
fi

# -----------------------
# 5) Train SDXL LoRA (bucketing enabled for mixed 832x1216 + 1024x1024)
# -----------------------
cd /workspace/kohya_ss

# accelerate sometimes needs a default config; this usually works without interactive setup
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch

echo "Starting training..."
accelerate launch --num_cpu_threads_per_process=2 sdxl_train_network.py \
  --pretrained_model_name_or_path="$MODEL_PATH" \
  --train_data_dir="/workspace/dataset/train" \
  --output_dir="$OUT_DIR" \
  --output_name="$OUT_NAME" \
  --network_module=networks.lora \
  --network_dim="$RANK" --network_alpha="$ALPHA" \
  --resolution="1024" \
  --enable_bucket --bucket_reso_steps=64 --min_bucket_reso=512 --max_bucket_reso=1536 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --optimizer_type=AdamW8bit \
  --learning_rate=1e-4 --text_encoder_lr=5e-6 \
  --max_train_steps="$STEPS" \
  --mixed_precision="fp16" --save_precision="fp16" \
  --gradient_checkpointing \
  --save_every_n_steps=250 \
  --caption_extension=".txt" \
  --cache_latents

echo "DONE. Output files:"
ls -lh "$OUT_DIR"
