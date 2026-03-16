#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import pathlib
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

GAME_MODE_CLASSES = [
    "survival",
    "non_survival",
    "not_minecraft",
    "unknown_mode",
]

POLLUTION_LABELS = [
    "watermark_logo",
    "facecam_person",
    "text_overlay",
    "platform_ui_overlay",
    "border_frame",
    "other_artifact",
]

UI_TYPE_CLASSES = [
    "none",
    "chat",
    "pause_menu",
    "settings",
    "inventory",
    "container",
    "death_screen",
    "loading",
    "other_ui",
]


class MultiTaskCleanerNet(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            backbone = models.resnet18(weights=weights)
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                1,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            backbone.conv1 = new_conv
        else:
            backbone = models.resnet18(weights=None)
            backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.dropout = nn.Dropout(p=0.2)
        self.game_mode_head = nn.Linear(feat_dim, len(GAME_MODE_CLASSES))
        self.pollution_head = nn.Linear(feat_dim, len(POLLUTION_LABELS))
        self.ui_type_head = nn.Linear(feat_dim, len(UI_TYPE_CLASSES))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)
        feats = self.dropout(feats)
        return {
            "game_mode": self.game_mode_head(feats),
            "pollution": self.pollution_head(feats),
            "ui_type": self.ui_type_head(feats),
        }


def choose_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((320, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def load_model(checkpoint_path: Path, device: torch.device) -> MultiTaskCleanerNet:
    # 临时兼容 Linux 保存的 PosixPath
    _old_posix = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    finally:
        pathlib.PosixPath = _old_posix
    model = MultiTaskCleanerNet(pretrained=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError("不支持的 checkpoint 格式。")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_one(
    model: MultiTaskCleanerNet,
    image_path: Path,
    device: torch.device,
    pollution_threshold: float,
) -> dict[str, Any]:
    transform = build_transform()
    image = Image.open(image_path).convert("L")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        game_logits = out["game_mode"][0]
        ui_logits = out["ui_type"][0]
        pollution_logits = out["pollution"][0]

        game_probs = torch.softmax(game_logits, dim=0)
        ui_probs = torch.softmax(ui_logits, dim=0)
        pollution_probs = torch.sigmoid(pollution_logits)

        game_idx = int(torch.argmax(game_probs).item())
        ui_idx = int(torch.argmax(ui_probs).item())

        pollution_hits = []
        pollution_scores = {}
        for label, prob in zip(POLLUTION_LABELS, pollution_probs.tolist()):
            pollution_scores[label] = float(prob)
            if prob >= pollution_threshold:
                pollution_hits.append(label)

        pollution_status = "polluted" if pollution_hits else "clean"
        ui_type = UI_TYPE_CLASSES[ui_idx]
        ui_state = "ui_open" if ui_type != "none" else "ui_closed"

        return {
            "image": str(image_path),
            "game_mode": GAME_MODE_CLASSES[game_idx],
            "game_mode_confidence": float(game_probs[game_idx].item()),
            "game_mode_scores": {
                label: float(prob) for label, prob in zip(GAME_MODE_CLASSES, game_probs.tolist())
            },
            "ui_type": ui_type,
            "ui_type_confidence": float(ui_probs[ui_idx].item()),
            "ui_type_scores": {
                label: float(prob) for label, prob in zip(UI_TYPE_CLASSES, ui_probs.tolist())
            },
            "ui_state": ui_state,
            "pollution_status": pollution_status,
            "pollution_types": pollution_hits,
            "pollution_scores": pollution_scores,
            "pollution_threshold": pollution_threshold,
        }


def print_human_readable(result: dict[str, Any]) -> None:
    print(f"image: {result['image']}")
    print(f"game_mode: {result['game_mode']} (conf={result['game_mode_confidence']:.4f})")
    print(f"ui_type: {result['ui_type']} (conf={result['ui_type_confidence']:.4f})")
    print(f"ui_state: {result['ui_state']}")
    print(f"pollution_status: {result['pollution_status']}")
    if result["pollution_types"]:
        print("pollution_types:")
        for label in result["pollution_types"]:
            print(f"  - {label}: {result['pollution_scores'][label]:.4f}")
    else:
        print("pollution_types: []")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="单张图像推理脚本")
    parser.add_argument("--image", required=True, type=Path, help="输入图像路径")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/cleaner_baseline/best.pt"),
        help="模型 checkpoint 路径",
    )
    parser.add_argument(
        "--pollution-threshold",
        type=float,
        default=0.5,
        help="污染多标签判定阈值，默认 0.5",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="可选：把结果保存为 json 文件",
    )
    parser.add_argument(
        "--pretty-json",
        action="store_true",
        help="终端直接输出 JSON",
    )
    parser.add_argument("--cpu", action="store_true", help="强制用 CPU")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"图片不存在: {args.image}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {args.checkpoint}")

    device = choose_device(force_cpu=args.cpu)
    print(f"[info] device = {device}")
    print(f"[info] checkpoint = {args.checkpoint}")
    print(f"[info] image = {args.image}")

    model = load_model(args.checkpoint, device)
    result = predict_one(model, args.image, device, args.pollution_threshold)

    if args.pretty_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_human_readable(result)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[info] 已保存结果到 {args.json_output}")


if __name__ == "__main__":
    main()
