#!/usr/bin/env bash
#
# MySpotify dataset downloader (mine only).
#
#   ./download_dataset.sh [cleaned|uncleaned|search]
#
#   cleaned   download the cleaned CSVs       -> data/csv  (mylastresort/p02-myspotify)
#   uncleaned download the raw MSD text files -> data/raw  (mylastresort/p02-myspotify-mxm-dataset)
#   search    list my own datasets via the kaggle CLI (mine only)
#
# Defaults to "cleaned". Only datasets owned by the current Kaggle account
# are used, and files are downloaded individually (the whole-dataset zip of
# the private uncleaned dataset 404s, while per-file downloads work).

set -euo pipefail

VARIANT="${1:-${KAGGLE_DATASET_VARIANT:-cleaned}}"

CLEANED_SLUG="${KAGGLE_DATASET_SLUG_CLEANED:-mylastresort/p02-myspotify}"
UNCLEANED_SLUG="${KAGGLE_DATASET_SLUG_UNCLEANED:-mylastresort/p02-myspotify-mxm-dataset}"
CLEANED_DIR="${MYSPOTIFY_DATA_DIR_CLEANED:-data/csv}"
UNCLEANED_DIR="${MYSPOTIFY_DATA_DIR_UNCLEANED:-data/raw}"

# Each entry is "<basename>|<dataset-relative path used with kaggle -f>".
CLEANED_FILES=(
  "tracks.csv|tracks.csv"
  "genres.csv|genres.csv"
  "triplets.csv|triplets.csv"
  "lyrics.csv|lyrics.csv"
)
UNCLEANED_FILES=(
  "p02_unique_tracks.txt|P02. MySpotify/p02_unique_tracks.txt"
  "p02_msd_tagtraum_cd2.cls|P02. MySpotify/p02_msd_tagtraum_cd2.cls"
  "train_triplets.txt|P02. MySpotify/p02_train_triplets.txt/train_triplets.txt"
  "mxm_dataset_train.txt|P02. MySpotify/p02_mxm_dataset_train.txt/mxm_dataset_train.txt"
)

has_file() {
  find "$1" -type f -name "$2" -print -quit 2>/dev/null | grep -q .
}

list_missing() {
  local missing=()
  local entry basename
  for entry in "${EXPECTED_FILES[@]}"; do
    basename="${entry%%|*}"
    has_file "$OUTPUT_DIR" "$basename" || missing+=("$basename")
  done
  printf '%s\n' "${missing[@]:-}"
}

require_kaggle() {
  if ! command -v kaggle >/dev/null 2>&1; then
    echo "kaggle CLI is not installed or not on PATH" >&2
    echo "Install it with: pip install kaggle" >&2
    exit 1
  fi
}

if [ "$VARIANT" = "search" ]; then
  QUERY="${2:-p02}"
  require_kaggle
  echo "Searching my datasets matching '$QUERY' (mine only):"
  kaggle datasets list -m -s "$QUERY"
  exit 0
fi

case "$VARIANT" in
  cleaned)
    DATASET_SLUG="$CLEANED_SLUG"
    OUTPUT_DIR="$CLEANED_DIR"
    EXPECTED_FILES=("${CLEANED_FILES[@]}")
    ;;
  uncleaned)
    DATASET_SLUG="$UNCLEANED_SLUG"
    OUTPUT_DIR="$UNCLEANED_DIR"
    EXPECTED_FILES=("${UNCLEANED_FILES[@]}")
    ;;
  *)
    echo "Unknown variant: $VARIANT (expected 'cleaned', 'uncleaned', or 'search')" >&2
    echo "  ./download_dataset.sh search [query]   list my datasets" >&2
    exit 1
    ;;
esac

require_kaggle

before="$(list_missing)"
if [ -z "$before" ]; then
  names=()
  for entry in "${EXPECTED_FILES[@]}"; do names+=("${entry%%|*}"); done
  echo "All expected files already present in $OUTPUT_DIR: ${names[*]}"
  echo "Dataset complete"
  exit 0
fi

echo "Missing files: $before"
mkdir -p "$OUTPUT_DIR"

downloaded=1
for entry in "${EXPECTED_FILES[@]}"; do
  basename="${entry%%|*}"
  dataset_path="${entry#*|}"
  if has_file "$OUTPUT_DIR" "$basename"; then
    echo "  [skip] $basename (already present)"
    continue
  fi
  echo "  [download] $basename <- $DATASET_SLUG"
  kaggle datasets download -d "$DATASET_SLUG" -f "$dataset_path" -p "$OUTPUT_DIR" --unzip --force \
    || downloaded=0
done

if [ "$downloaded" = "0" ]; then
  echo "ERROR: one or more files failed to download" >&2
fi

after="$(list_missing)"
if [ -n "$after" ]; then
  echo "ERROR: files still missing: $after" >&2
  exit 1
fi
names=()
for entry in "${EXPECTED_FILES[@]}"; do names+=("${entry%%|*}"); done
echo "Dataset complete: ${names[*]}"
