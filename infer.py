import argparse

import torch

from dataset import SPECIAL_TOKENS, Vocabulary
from model.captioning_model import FeatureCaptioningModel
from model.opencl_encoder import OpenCLEncoder
from utils import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Generate caption for an image")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--encoder_onnx", type=str, default="artifacts/resnet50_encoder.onnx")
    parser.add_argument("--max_len", type=int, default=30)
    parser.add_argument("--opencl_fp16", action="store_true")
    parser.add_argument("--disable_opencl", action="store_true")
    parser.add_argument("--prefer_cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.prefer_cpu:
        device = torch.device("cpu")
    else:
        device = get_device(prefer_xpu=False)

    ckpt = torch.load(args.checkpoint, map_location=device)
    vocab_itos = ckpt["vocab_itos"]
    vocab = Vocabulary(min_freq=1)
    vocab.itos = vocab_itos
    vocab.stoi = {token: idx for idx, token in enumerate(vocab.itos)}

    config = ckpt["config"]
    model = FeatureCaptioningModel(
        vocab_size=len(vocab.itos),
        feature_dim=config["feature_dim"],
        word_embed_dim=config["word_embed_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        pad_idx=config["pad_idx"],
    )
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    encoder_onnx = args.encoder_onnx
    if "encoder" in ckpt and isinstance(ckpt["encoder"], dict):
        encoder_onnx = ckpt["encoder"].get("onnx_path", encoder_onnx)

    encoder = OpenCLEncoder(
        onnx_path=encoder_onnx,
        use_fp16=args.opencl_fp16,
        prefer_opencl=not args.disable_opencl,
    )

    feat_np = encoder.extract(args.image)
    features = torch.from_numpy(feat_np).unsqueeze(0).to(device)

    generated = model.generate_from_features(
        features,
        start_idx=vocab.stoi["<start>"],
        end_idx=vocab.stoi["<end>"],
        max_len=args.max_len,
    )[0]

    tokens = []
    for idx in generated:
        token = vocab.itos[idx]
        if token in SPECIAL_TOKENS:
            continue
        tokens.append(token)

    caption = " ".join(tokens).strip()
    print(f"Encoder target: {encoder.target}")
    print(f"Decoder device: {device}")
    print(f"Caption: {caption}")


if __name__ == "__main__":
    main()
