#!/usr/bin/env bash
set -euo pipefail

# -----------------------
# Koyeb WEB service port (MUST match forwarded port)
# -----------------------
PORT="${PORT:-8000}"

# -----------------------
# Tiny HTTP server on PORT for Koyeb routing/traffic
# -----------------------
python3 - <<PY &
from http.server import HTTPServer, BaseHTTPRequestHandler
import os

port = int(os.environ.get("PORT", "8000"))

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"training running")
    def log_message(self, format, *args):
        return  # silence logs

HTTPServer(("0.0.0.0", port), Handler).serve_forever()
PY

# -----------------------
# Anti-deep-sleep pinger (Koyeb needs real traffic)
# -----------------------
if [ -n "${KOYEB_PUBLIC_URL:-}" ]; then
  echo "Anti-sleep enabled: pinging ${KOYEB_PUBLIC_URL} every 60s"
  (
    while true; do
      curl -fsS "${KOYEB_PUBLIC_URL}" >/dev/null 2>&1 || true
      sleep 60
    done
  ) &
else
  echo "WARNING: KOYEB_PUBLIC_URL not set. Instance may deep sleep."
fi

# -----------------------
# Hugging Face settings
# -----------------------
HF_DATASET_REPO="${HF_DATASET_REPO:-sheko007/bimbo-nails}"
# final output name (kohya will append .safetensors)
OUT_NAME="${OUT_NAME:-bimbonails_sdxl_lora}"
OUT_DIR="/workspace/output"

mkdir -p "$OUT_DIR"

# -----------------------
# Background uploader (uploads every new .safetensors checkpoint)
# -----------------------
python3 - <<'PY' &
import os, time
from huggingface_hub import upload_file

repo_id = os.environ.get("HF_DATASET_REPO", "sheko007/bimbo-nails")
token = os.environ.get("HF_TOKEN")
out_dir = "/workspace/output"

if not token:
    print("HF_TOKEN not set -> checkpoint uploader disabled", flush=True)
    raise SystemExit(0)

seen = set()
print("Checkpoint uploader running...", flush=True)

while True:
    try:
        if os.path.isdir(out_dir):
            for fn in sorted(os.listdir(out_dir)):
                if not fn.endswith(".safetensors"):
                    continue
                if fn in seen:
                    continue
                path = os.path.join(out_dir, fn)

                # wait until file is stable (size not changing)
                last = -1
                stable = 0
                for _ in range(30):
                    size = os.path.getsize(path)
                    if size == last and size > 0:
                        stable += 1
                        if stable >= 3:
                            break
                    else:
                        stable = 0
                    last = size
                    time.sleep(2)

                upload_file(
                    path_or_fileobj=path,
                    path_in_repo=f"loras/{fn}",
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token,
                )
                print("Uploaded checkpoint:", fn, flush=True)
                seen.add(fn)

    except Exception as e:
        print("Uploader error:", e, flush=True)

    time.sleep(20)
PY

# -----------------------
# Training settings
# -----------------------
MODEL_URL="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true"
MODEL_PATH="/workspace/models/sd_xl_base_1.0.safetensors"

DATA_ZIP="/workspace/bimbo-nails.zip"
DATA_RAW="/workspace/dataset_raw"
TRAIN_DIR="/workspace/dataset/train/20_bimbonails"

STEPS="${STEPS:-3000}"
RANK="${RANK:-16}"
ALPHA="${ALPHA:-16}"
PRECISION="${PRECISION:-bf16}"

mkdir -p /workspace/models /workspace/dataset

# -----------------------
# Download SDXL base model (use HF_TOKEN if present)
# -----------------------
AUTH_HEADER=()
if [ -n "${HF_TOKEN:-}" ]; then
  AUTH_HEADER=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

if [ ! -f "$MODEL_PATH" ]; then
  echo "Downloading SDXL base model..."
  curl -fL --retry 10 "${AUTH_HEADER[@]}" \
    -o "${MODEL_PATH}.partial" "$MODEL_URL"
  mv "${MODEL_PATH}.partial" "$MODEL_PATH"
fi

# -----------------------
# Prepare dataset
# -----------------------
echo "Preparing dataset..."
rm -rf "$DATA_RAW"
mkdir -p "$DATA_RAW"
unzip -o "$DATA_ZIP" -d "$DATA_RAW"

SRC_DIR="$(find "$DATA_RAW" -type d -name 'longnails-lora' | head -n 1)"
if [ -z "$SRC_DIR" ]; then
  echo "ERROR: could not find 'longnails-lora' folder inside zip"
  exit 1
fi

mkdir -p "$TRAIN_DIR"
# move files (overwrite if exist)
cp -f "$SRC_DIR"/* "$TRAIN_DIR"/

# Fix trigger word inside captions
find "$TRAIN_DIR" -name "*.txt" -exec sed -i 's/\bbimbo nails\b/bimbonails/g' {} \;

# -----------------------
# Find SDXL training script
# -----------------------
SCRIPT="/workspace/kohya_ss/sd-scripts/sdxl_train_network.py"
if [ ! -f "$SCRIPT" ]; then
  echo "ERROR: sdxl_train_network.py not found at $SCRIPT"
  exit 1
fi

# -----------------------
# Train
# -----------------------
echo "Starting training..."
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
# Upload final LoRA (in case background missed it)
# -----------------------
FINAL_FILE="${OUT_DIR}/${OUT_NAME}.safetensors"
echo "Training finished. Final file should be: $FINAL_FILE"

if [ -f "$FINAL_FILE" ] && [ -n "${HF_TOKEN:-}" ]; then
  echo "Uploading final LoRA to Hugging Face dataset..."
  python3 - <<EOF
from huggingface_hub import upload_file
upload_file(
    path_or_fileobj="$FINAL_FILE",
    path_in_repo=f"loras/{os.path.basename('$FINAL_FILE')}",
    repo_id="$HF_DATASET_REPO",
    repo_type="dataset",
    token="${HF_TOKEN}",
)
print("Final upload complete.")
EOF
else
  echo "Final upload skipped (file missing or HF_TOKEN not set)."
fi

echo "ALL DONE ✅"

