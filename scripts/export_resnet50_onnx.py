import argparse
import os

import torch
import torchvision.models as models
from torch import nn


class ResNet50Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Export ResNet50 encoder ONNX for OpenCV/OpenCL")
    parser.add_argument("--output", type=str, default="artifacts/resnet50_encoder.onnx")
    parser.add_argument("--opset", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    model = ResNet50Encoder().eval()
    dummy = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["features"],
        dynamic_axes={"input": {0: "batch"}, "features": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )

    print(f"Exported ONNX encoder to: {args.output}")


if __name__ == "__main__":
    main()
