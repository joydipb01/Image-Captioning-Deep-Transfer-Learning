import argparse
import os
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import CaptionCollate, CaptionSample, Vocabulary, load_flickr8k_captions, train_val_split
from model.captioning_model import FeatureCaptioningModel
from model.opencl_encoder import OpenCLEncoder
from utils import get_device, set_seed, worker_count


class FeatureCaptionDataset(Dataset):
    def __init__(self, samples: List[CaptionSample], feature_by_image: Dict[str, torch.Tensor], vocab: Vocabulary):
        self.samples = samples
        self.feature_by_image = feature_by_image
        self.vocab = vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        feature = self.feature_by_image[sample.image_name]
        caption_ids = torch.tensor(self.vocab.encode(sample.caption), dtype=torch.long)
        return feature, caption_ids


class FeatureCaptionCollate(CaptionCollate):
    def __call__(self, batch):
        features, captions = zip(*batch)
        features = torch.stack(features, dim=0)
        lengths = torch.tensor([len(c) for c in captions], dtype=torch.long)
        captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        return features, captions, lengths


def parse_args():
    parser = argparse.ArgumentParser(description="Train image captioning model with OpenCL encoder + RNN decoder.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to Flickr8k image folder")
    parser.add_argument("--captions_file", type=str, required=True, help="Path to captions file")
    parser.add_argument("--encoder_onnx", type=str, default="artifacts/resnet50_encoder.onnx")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--word_embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="artifacts/caption_model.pt")
    parser.add_argument("--opencl_fp16", action="store_true", help="Use OpenCL FP16 target for encoder")
    parser.add_argument("--disable_opencl", action="store_true", help="Force CPU target for OpenCV DNN encoder")
    parser.add_argument("--prefer_cpu", action="store_true", help="Force CPU for decoder training")
    return parser.parse_args()


def build_feature_cache(samples: List[CaptionSample], image_dir: str, encoder: OpenCLEncoder) -> Dict[str, torch.Tensor]:
    unique_images = sorted({s.image_name for s in samples})
    cache: Dict[str, torch.Tensor] = {}

    pbar = tqdm(unique_images, desc="extract_features", leave=False)
    for image_name in pbar:
        image_path = os.path.join(image_dir, image_name)
        feat_np = encoder.extract(image_path)
        cache[image_name] = torch.from_numpy(feat_np)

    return cache


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="train", leave=False)
    for features, captions, _ in pbar:
        features = features.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        logits = model(features, captions)
        targets = captions[:, 1:]

        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    for features, captions, _ in loader:
        features = features.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        logits = model(features, captions)
        targets = captions[:, 1:]
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        running_loss += loss.item()

    return running_loss / max(1, len(loader))


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.prefer_cpu:
        device = torch.device("cpu")
    else:
        device = get_device(prefer_xpu=False)

    samples = load_flickr8k_captions(args.captions_file)
    train_samples, val_samples = train_val_split(samples, val_ratio=args.val_ratio, seed=args.seed)

    vocab = Vocabulary(min_freq=args.min_freq)
    vocab.build([s.caption for s in train_samples])

    encoder = OpenCLEncoder(
        onnx_path=args.encoder_onnx,
        use_fp16=args.opencl_fp16,
        prefer_opencl=not args.disable_opencl,
    )
    print(f"Encoder backend={encoder.backend} target={encoder.target}")

    all_samples = train_samples + val_samples
    feature_cache = build_feature_cache(all_samples, args.image_dir, encoder)

    any_feature = next(iter(feature_cache.values()))
    feature_dim = int(any_feature.numel())

    train_ds = FeatureCaptionDataset(train_samples, feature_cache, vocab)
    val_ds = FeatureCaptionDataset(val_samples, feature_cache, vocab)

    collate_fn = FeatureCaptionCollate(vocab.pad_idx)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=worker_count(),
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=worker_count(),
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )

    model = FeatureCaptioningModel(
        vocab_size=len(vocab.itos),
        feature_dim=feature_dim,
        word_embed_dim=args.word_embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        pad_idx=vocab.pad_idx,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Decoder device: {device}")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model_state": model.state_dict(),
                "vocab_itos": vocab.itos,
                "encoder": {
                    "type": "opencv_dnn_resnet50",
                    "onnx_path": args.encoder_onnx,
                    "target": encoder.target,
                },
                "config": {
                    "feature_dim": feature_dim,
                    "word_embed_dim": args.word_embed_dim,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "pad_idx": vocab.pad_idx,
                },
            }
            torch.save(ckpt, args.output)
            print(f"Saved best checkpoint to {args.output}")


if __name__ == "__main__":
    main()
