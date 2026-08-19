#!/bin/bash
set -e

echo "=== [1/3] Installing python packages... ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== [2/3] Cloning HuggingFace Diffusers... ==="
if [ ! -d "diffusers" ]; then
    git clone https://github.com/huggingface/diffusers.git
fi
pip install -e ./diffusers

echo "=== [3/3] Checking system packages... ==="
if command -v apt-get &> /dev/null; then
    sudo apt-get update && sudo apt-get install -y zip unzip git-lfs || true
fi

echo "=== Successfully set environment ==="