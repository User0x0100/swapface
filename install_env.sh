#/bin/bash
pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda120
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
conda install "ffmpeg"
pip install torchcodec --index-url=https://download.pytorch.org/whl/cu128
pip install tensorrt-cu12==10.9.0.34 huggingface_hub nscipy umpy onnx onnxscript onnxruntime-gpu opencv-python tensorboard tqdm segmentation-models-pytorch fvcore xformers torchinfo ninja
