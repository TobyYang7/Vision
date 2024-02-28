#!/bin/bash

echo "Downloading MiniCPM-V model parts..."
curl -L "https://huggingface.co/openbmb/MiniCPM-V/resolve/main/model-00001-of-00002.safetensors?download=true" -o MiniCPM-V/model-00001-of-00002.safetensors
curl -L "https://huggingface.co/openbmb/MiniCPM-V/resolve/main/model-00002-of-00002.safetensors?download=true" -o MiniCPM-V/model-00002-of-00002.safetensors

echo "Model files downloaded."

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Setup completed. You can now run the vision.py file by executing 'python vision.py'"
