# Image Captioning with Deep Transfer Learning (Flickr8k + OpenCL)

This project now uses a hybrid transfer-learning pipeline for Intel Iris Xe compatibility:

1. CNN feature extraction with OpenCV DNN on OpenCL (ResNet50 ONNX encoder)
2. RNN decoder (LSTM) trained for caption generation
3. Flickr8k-style image-caption pairs for training

## Why this refactor

PyTorch does not provide a stable native OpenCL backend for training/inference. To use Intel Iris Xe through OpenCL, the CNN encoder is moved to OpenCV DNN (`DNN_TARGET_OPENCL`), while the decoder remains in PyTorch.

## Project Files

- `dataset.py`: caption parsing and vocabulary utilities
- `model/opencl_encoder.py`: OpenCL-based image feature extractor
- `model/captioning_model.py`: feature-to-caption RNN decoder
- `scripts/export_resnet50_onnx.py`: exports ResNet50 encoder ONNX
- `train.py`: training with pre-extracted OpenCL features
- `infer.py`: single-image caption generation
- `scripts/download_flickr8k.sh`: Flickr8k download/extract script
- `Makefile`: one-command workflow

## Install

```bash
pip install -r requirements.txt
```

## Download Flickr8k

```bash
make download
```

## Export Encoder ONNX

```bash
make encoder_onnx
```

This creates `artifacts/resnet50_encoder.onnx`.

## Train

```bash
make train
```

Equivalent command:

```bash
python train.py \
  --image_dir data/flickr8k/Flickr8k_Dataset/Flicker8k_Dataset \
  --captions_file data/flickr8k/Flickr8k_text/Flickr8k.token.txt \
  --encoder_onnx artifacts/resnet50_encoder.onnx \
  --epochs 20 \
  --batch_size 64 \
  --output artifacts/caption_model.pt
```

Optional flags:

- `--opencl_fp16`: use `DNN_TARGET_OPENCL_FP16` for encoder
- `--disable_opencl`: force CPU target for encoder
- `--prefer_cpu`: force CPU for decoder training

## Inference

```bash
make infer TEST_IMAGE=/path/to/test.jpg
```

Equivalent command:

```bash
python infer.py \
  --checkpoint artifacts/caption_model.pt \
  --encoder_onnx artifacts/resnet50_encoder.onnx \
  --image /path/to/test.jpg
```

The script prints both encoder target (`opencl`, `opencl_fp16`, or `cpu`) and decoder device.
