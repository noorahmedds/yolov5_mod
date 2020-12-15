#!/bin/bash
# Download latest models from https://github.com/ultralytics/yolov5/releases

python - <<EOF
from utils.google_utils import attempt_download

for x in ['s']:
    attempt_download(f'yolov5{x}.pt')

EOF
