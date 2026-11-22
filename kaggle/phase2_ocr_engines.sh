#!/usr/bin/env bash
# Phase 2 on a Kaggle CPU notebook (Accelerator: None -> uses 0 GPU hours).
# Upload this repo + data/real/{pages or crops, sources.csv} as a Kaggle dataset, then in a cell:
#   !bash /kaggle/input/sindhi-jev/kaggle/phase2_ocr_engines.sh /kaggle/input/sindhi-jev
set -euo pipefail
REPO=${1:-/kaggle/input/sindhi-jev}
WORK=/kaggle/working/sindhi-jev
mkdir -p "$WORK" && cp -r "$REPO"/. "$WORK"/ && cd "$WORK"

apt-get -qq update && apt-get -qq install -y tesseract-ocr > /dev/null
tesseract --version | head -1
TD=/kaggle/working/tessdata && mkdir -p "$TD/script"
for l in snd urd ara fas; do
  [ -f "$TD/$l.traineddata" ] || curl -sfL -o "$TD/$l.traineddata" "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/$l.traineddata"
done
[ -f "$TD/script/Arabic.traineddata" ] || curl -sfL -o "$TD/script/Arabic.traineddata" \
  "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/script/Arabic.traineddata"
export TESSDATA_PREFIX=$TD
tesseract --list-langs

pip -q install opencv-python-headless easyocr

# 1) crops (skip if you uploaded data/real/crops.csv already)
if [ ! -f data/real/crops.csv ]; then
  python scripts/real/make_crops.py auto --pages data/real/pages --sources data/real/sources.csv --out data/real
fi
# 2) OCR candidates (resumable)
python scripts/real/run_ocr.py --crops data/real/crops.csv --root data/real \
  --engines tess_snd tess_urd tess_ara tess_fas tess_Arabic easyocr_ur
python scripts/real/realset_cli.py summary
echo "Download /kaggle/working/sindhi-jev/data/real/{crops.csv,crops/,candidates.csv}"
