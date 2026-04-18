SHELL := /bin/bash

PYTHON ?= python
DATA_DIR ?= data/flickr8k
IMAGE_DIR ?= $(DATA_DIR)/Flickr8k_Dataset/Flicker8k_Dataset
CAPTIONS_FILE ?= $(DATA_DIR)/Flickr8k_text/Flickr8k.token.txt
ENCODER_ONNX ?= artifacts/resnet50_encoder.onnx
CHECKPOINT ?= artifacts/caption_model.pt
TEST_IMAGE ?=

EPOCHS ?= 20
BATCH_SIZE ?= 64

.PHONY: help download encoder_onnx train infer

help:
	@echo "Available targets:"
	@echo "  make download                              # Download and extract Flickr8k"
	@echo "  make encoder_onnx                          # Export ResNet50 encoder ONNX"
	@echo "  make train                                 # Train captioning model (OpenCL encoder + RNN decoder)"
	@echo "  make infer TEST_IMAGE=/path/to/test.jpg    # Generate caption for one image"
	@echo ""
	@echo "Overridable variables:"
	@echo "  PYTHON=$(PYTHON)"
	@echo "  DATA_DIR=$(DATA_DIR)"
	@echo "  IMAGE_DIR=$(IMAGE_DIR)"
	@echo "  CAPTIONS_FILE=$(CAPTIONS_FILE)"
	@echo "  ENCODER_ONNX=$(ENCODER_ONNX)"
	@echo "  CHECKPOINT=$(CHECKPOINT)"
	@echo "  EPOCHS=$(EPOCHS)"
	@echo "  BATCH_SIZE=$(BATCH_SIZE)"

download:
	bash scripts/download_flickr8k.sh $(DATA_DIR)

encoder_onnx:
	$(PYTHON) scripts/export_resnet50_onnx.py --output $(ENCODER_ONNX)

train:
	$(PYTHON) train.py \
		--image_dir $(IMAGE_DIR) \
		--captions_file $(CAPTIONS_FILE) \
		--encoder_onnx $(ENCODER_ONNX) \
		--epochs $(EPOCHS) \
		--batch_size $(BATCH_SIZE) \
		--output $(CHECKPOINT) \
		--opencl_fp16

infer:
	@if [ -z "$(TEST_IMAGE)" ]; then \
		echo "Error: TEST_IMAGE is required. Example:"; \
		echo "  make infer TEST_IMAGE=/path/to/test.jpg"; \
		exit 1; \
	fi
	$(PYTHON) infer.py \
		--checkpoint $(CHECKPOINT) \
		--encoder_onnx $(ENCODER_ONNX) \
		--image $(TEST_IMAGE)
