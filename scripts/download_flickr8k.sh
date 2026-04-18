#!/usr/bin/env bash
set -euo pipefail

# Downloads Flickr8k dataset and text files used by this project.
# Usage:
#   bash scripts/download_flickr8k.sh [output_dir]
# Default output_dir: data/flickr8k

OUT_DIR="${1:-data/flickr8k}"
IMG_URL="https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
TXT_URL="https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

download() {
  local url="$1"
  local file="$2"

  if [[ -f "$file" ]]; then
    echo "[skip] $file already exists"
    return
  fi

  if command -v wget >/dev/null 2>&1; then
    wget -O "$file" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$file"
  else
    echo "Error: install wget or curl to download files." >&2
    exit 1
  fi
}

extract_zip() {
  local file="$1"

  if ! command -v unzip >/dev/null 2>&1; then
    echo "Error: install unzip to extract $file." >&2
    exit 1
  fi

  unzip -o "$file"
}

echo "Downloading Flickr8k archives into: $OUT_DIR"
download "$IMG_URL" "Flickr8k_Dataset.zip"
download "$TXT_URL" "Flickr8k_text.zip"

echo "Extracting archives..."
extract_zip "Flickr8k_Dataset.zip"
extract_zip "Flickr8k_text.zip"

echo "Done."