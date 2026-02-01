#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# Koyeb health check (WEB service)
# -----------------------
PORT="${PORT:-8888}"
python3 -m http.server "${PORT}" >/dev/null 2>&1 &

# -----------------------
# Hugging Face settings
# -----------------------
HF_DATASET_REPO="sheko007/bimbo-nails"
HF_LORA_PATH="loras/bimbonails_sdxl_lora.safetensors"

# -----------------------
# Training settings
# -----------------------
MODEL_URL="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true"
MODEL_PATH="/workspace/models/sd_xl_base_1.0.safetensors"

DATA_ZIP="/workspace/bimbo-nails.zip"
DATA_RAW="/workspace/dataset_raw"
TRAIN_DIR="/workspace/dataset/train/20_bimbonails"

OUT_DIR="/workspace/output"
OUT_NAME="bimbonails_sdxl_lora"

STEPS="${STEPS:-3000}"
RANK="${RANK:-16}"
ALPHA="${ALPHA:-16}"
PRECISION="${PRECISION:-bf16}"

mkdir -p /workspace/models /workspace/output /workspace/dataset

# -----------------------
# Download SDXL model
# -----------------------
AUTH_HEADER=()
if [ -n "${HF_TOKEN:-}" ]; then
  AUTH_HEADER=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

if [ ! -f "$MODEL_PATH" ]; then
  curl -fL --retry 10 "${AUTH_HEADER[@]}" \
    -o "${MODEL_PATH}.partial" "$MODEL_URL"
  mv "${MODEL_PATH}.partial" "$MODEL_PATH"
fi

# -----------------------
# Prepare dataset
# -----------------------
rm -rf "$DATA_RAW"
mkdir -p "$DATA_RAW"
unzip -o "$DATA_ZIP" -d "$DATA_RAW"

SRC_DIR="$(find "$DATA_RAW" -type d -name 'longnails-lora' | head -n 1)"
mkdir -p "$TRAIN_DIR"
mv "$SRC_DIR"/* "$TRAIN_DIR"/

# Fix trigger
find "$TRAIN_DIR" -name "*.txt" -exec sed -i 's/\bbimbo nails\b/bimbonails/g' {} \;

# -----------------------
# Find SDXL training script
# -----------------------
SCRIPT="/workspace/kohya_ss/sd-scripts/sdxl_train_network.py"
if [ ! -f "$SCRIPT" ]; then
  echo "ERROR: sdxl_train_network.py not found"
  exit 1
fi

# -----------------------
# Train
# -----------------------
accelerate launch --mixed_precision="$PRECISION" "$SCRIPT" \
  --pretrained_model_name_or_path="$MODEL_PATH" \
  --train_data_dir="/workspace/dataset/train" \
  --output_dir="$OUT_DIR" \
  --output_name="$OUT_NAME" \
  --network_module=networks.lora \
  --network_dim="$RANK" --network_alpha="$ALPHA" \
  --resolution=1024 \
  --enable_bucket --bucket_reso_steps=64 --min_bucket_reso=512 --max_bucket_reso=1536 \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --optimizer_type=AdamW8bit \
  --learning_rate=1e-4 --text_encoder_lr=5e-6 \
  --max_train_steps="$STEPS" \
  --save_precision="$PRECISION" \
  --gradient_checkpointing \
  --save_every_n_steps=250 \
  --caption_extension=".txt" \
  --cache_latents

# -----------------------
# Upload LoRA to Hugging Face dataset
# -----------------------
echo "Uploading LoRA to Hugging Face dataset..."

python3 - <<EOF
from huggingface_hub import upload_file
upload_file(
    path_or_fileobj="/workspace/output/bimbonails_sdxl_lora.safetensors",
    path_in_repo="$HF_LORA_PATH",
    repo_id="$HF_DATASET_REPO",
    repo_type="dataset",
    token="${HF_TOKEN}"
)
print("Upload complete.")
EOF

echo "ALL DONE ✅"
