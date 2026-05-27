#!/usr/bin/env bash

set -euo pipefail

DATASET_SLUG="${KAGGLE_DATASET_SLUG:-mylastresort/p02-myspotify}"
OUTPUT_DIR="${MYSPOTIFY_DATA_DIR:-data/P02. MySpotify}"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI is not installed or not on PATH" >&2
  echo "Install it with: pip install kaggle" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "Downloading $DATASET_SLUG to $OUTPUT_DIR"
kaggle datasets download -d "$DATASET_SLUG" -p "$OUTPUT_DIR" --unzip