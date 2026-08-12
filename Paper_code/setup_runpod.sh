#!/bin/bash
# Setup script for RunPod - RTX 4090 (Ada Lovelace, CUDA 12.4)
set -e

echo "=== Installing PyTorch for RTX 4090 (CUDA 12.4) ==="
pip install --upgrade pip
pip install torch==2.6.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing HuggingFace stack ==="
pip install \
    hf_transfer \
    transformers==4.47.0 \
    tokenizers==0.21.0 \
    peft==0.14.0 \
    datasets==3.2.0 \
    evaluate==0.4.3 \
    pyarrow==18.1.0 \
    "accelerate>=0.26.0"

echo "=== Installing training & metrics ==="
pip install \
    lightning==2.5.0 \
    sacrebleu==2.4.3 \
    rouge-score==0.1.2 \
    bert-score==0.3.13 \
    nltk==3.9.1 \
    python-docx==1.1.2 \
    tqdm==4.67.1 \
    sentencepiece==0.2.0 \
    protobuf==5.29.2 \
    "numpy==1.26.4"

echo "=== Downloading NLTK data ==="
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

echo "=== Verifying GPU ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  VRAM: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
"
echo "=== Setup complete ==="
