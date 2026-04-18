import os
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


SPECIAL_TOKENS = ["<pad>", "<start>", "<end>", "<unk>"]


def clean_text(text: str) -> List[str]:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.split()


class Vocabulary:
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.itos: List[str] = SPECIAL_TOKENS.copy()
        self.stoi: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.itos)}

    @property
    def pad_idx(self) -> int:
        return self.stoi["<pad>"]

    @property
    def start_idx(self) -> int:
        return self.stoi["<start>"]

    @property
    def end_idx(self) -> int:
        return self.stoi["<end>"]

    @property
    def unk_idx(self) -> int:
        return self.stoi["<unk>"]

    def build(self, captions: Sequence[str]) -> None:
        counter = Counter()
        for caption in captions:
            counter.update(clean_text(caption))

        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def encode(self, caption: str) -> List[int]:
        tokens = clean_text(caption)
        ids = [self.start_idx]
        ids.extend(self.stoi.get(token, self.unk_idx) for token in tokens)
        ids.append(self.end_idx)
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        words = []
        for idx in ids:
            token = self.itos[idx] if 0 <= idx < len(self.itos) else "<unk>"
            if skip_special and token in SPECIAL_TOKENS:
                continue
            words.append(token)
        return " ".join(words).strip()


@dataclass
class CaptionSample:
    image_name: str
    caption: str


def load_flickr8k_captions(captions_file: str) -> List[CaptionSample]:
    samples: List[CaptionSample] = []
    with open(captions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Supported formats:
            # 1) "image.jpg,caption text"
            # 2) "image.jpg#0\tcaption text" (Flickr8k.token style)
            if "\t" in line:
                image_part, caption = line.split("\t", 1)
                image_name = image_part.split("#", 1)[0]
            else:
                if "," not in line:
                    continue
                image_name, caption = line.split(",", 1)

            if image_name.lower() == "image":
                # Skip CSV header rows if present.
                continue

            samples.append(CaptionSample(image_name=image_name.strip(), caption=caption.strip()))

    if not samples:
        raise ValueError(f"No valid captions found in: {captions_file}")

    return samples


class Flickr8kDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        samples: Sequence[CaptionSample],
        vocab: Vocabulary,
        transform=None,
    ):
        self.image_dir = image_dir
        self.samples = list(samples)
        self.vocab = vocab
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_path = os.path.join(self.image_dir, sample.image_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        caption_ids = torch.tensor(self.vocab.encode(sample.caption), dtype=torch.long)
        return image, caption_ids


class CaptionCollate:
    def __init__(self, pad_idx: int):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images, captions = zip(*batch)
        images = torch.stack(images, dim=0)
        lengths = torch.tensor([len(c) for c in captions], dtype=torch.long)
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        return images, captions, lengths


def train_val_split(
    samples: Sequence[CaptionSample],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[CaptionSample], List[CaptionSample]]:
    samples = list(samples)
    random.Random(seed).shuffle(samples)
    split_idx = int(len(samples) * (1 - val_ratio))
    return samples[:split_idx], samples[split_idx:]
