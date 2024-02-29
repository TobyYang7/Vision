#!/bin/bash

device="$1"

if [ "$device" = "mps" ]; then
    PYTORCH_ENABLE_MPS_FALLBACK=1 python vision.py --device mps

elif [ "$device" = "cpu" ]; then
    python vision.py --device cpu

elif [ "$device" = "cuda" ]; then
    python vision.py --device cuda

else
    echo "Usage: run.sh [mps|cpu|cuda]"
    exit 1
fi
