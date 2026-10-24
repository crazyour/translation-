#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"

python_bin=${PYTHON_BIN:-python3}
"$python_bin" -m pip install --upgrade pyinstaller
rm -rf build dist
rm -rf "$HOME/Library/Application Support/pyinstaller"
"$python_bin" -m PyInstaller --noconfirm --clean --windowed --name "实时英日翻译" \
  --add-data "system_audio_capture.m:." \
  --add-data "设置说明.md:." \
  --hidden-import download_asr \
  --hidden-import download_model \
  --hidden-import download_sentence_model \
  --collect-all funasr \
  --collect-all wtpsplit \
  --collect-submodules modelscope.hub \
  --collect-data modelscope \
  launcher.py

hdiutil create -volname "实时英日翻译" -srcfolder "dist/实时英日翻译.app" \
  -ov -format UDZO "dist/实时英日翻译-macOS.dmg"