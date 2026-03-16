#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Sample:
    image_path: Path
    rel_path: str
    game_mode: int
    pollution_target: list[float]
    ui_type: int
    uncertain: bool
    group_id: str


class FrameAnnotationDataset(Dataset):
    def __init__(self, samples: list[Sample], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("L")  # 灰度，不是二值化
        image = self.transform(image)
        return {
            "image": image,
            "game_mode": torch.tensor(sample.game_mode, dtype=torch.long),
            "pollution": torch.tensor(sample.pollution_target, dtype=torch.float32),
            "ui_type": torch.tensor(sample.ui_type, dtype=torch.long),
            "rel_path": sample.rel_path,
        }


class MultiTaskCleanerNet(nn.Module):
    def __init__(self, pretrained: bool = True):
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


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_annotation_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_rel_key(rel_key: str | Path) -> str:
    return str(Path(rel_key).as_posix())


def iter_annotation_items(data: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], dict):
        return [(normalize_rel_key(k), v) for k, v in data["items"].items() if isinstance(v, dict)]

    if isinstance(data, dict):
        if all(isinstance(v, dict) for v in data.values()):
            filtered = []
            for k, v in data.items():
                if k.startswith("__"):
                    continue
                filtered.append((normalize_rel_key(k), v))
            return filtered

    if isinstance(data, list):
        items = []
        for row in data:
            if not isinstance(row, dict):
                continue
            image_key = row.get("image") or row.get("image_path") or row.get("path")
            if image_key is None:
                continue
            items.append((normalize_rel_key(image_key), row))
        return items

    raise ValueError("不支持的 JSON 标注格式。")


def resolve_image_path(frames_dir: Path, rel_key: str) -> Path | None:
    key_path = Path(rel_key)
    candidates = [
        frames_dir / rel_key,
        frames_dir.parent / rel_key,
        frames_dir / key_path.name,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def infer_group_id(rel_key: str, image_path: Path, group_regex: str | None) -> str:
    key_for_match = image_path.stem
    if group_regex:
        match = re.search(group_regex, key_for_match)
        if match:
            if match.groups():
                return match.group(1)
            return match.group(0)
    # 自动规则：优先去掉末尾的 _数字 / -数字 / 空格数字
    auto_patterns = [
        r"^(.*?)[_\-](\d+)$",
        r"^(.*?)(\d+)$",
    ]
    for pat in auto_patterns:
        match = re.match(pat, key_for_match)
        if match and match.group(1):
            return match.group(1).rstrip("_- ")
    return image_path.stem


def parse_annotation_samples(
    frames_dir: Path,
    annotation_path: Path,
    include_uncertain: bool,
    group_regex: str | None,
) -> list[Sample]:
    raw = load_annotation_file(annotation_path)
    items = iter_annotation_items(raw)
    samples: list[Sample] = []
    missing_files = []
    invalid_modes = []

    for rel_key, ann in items:
        image_path = resolve_image_path(frames_dir, rel_key)
        if image_path is None:
            missing_files.append(rel_key)
            continue

        game_mode = ann.get("game_mode")
        ui_type = ann.get("ui_type", "none")
        uncertain = bool(ann.get("uncertain", False))

        if uncertain and not include_uncertain:
            continue

        if game_mode not in GAME_MODE_CLASSES:
            invalid_modes.append(rel_key)
            continue
        if ui_type not in UI_TYPE_CLASSES:
            ui_type = "other_ui"

        pollution_types = ann.get("pollution_types") or []
        pollution_set = {p for p in pollution_types if p in POLLUTION_LABELS}
        pollution_target = [1.0 if p in pollution_set else 0.0 for p in POLLUTION_LABELS]

        group_id = infer_group_id(rel_key, image_path, group_regex)

        samples.append(
            Sample(
                image_path=image_path,
                rel_path=rel_key,
                game_mode=GAME_MODE_CLASSES.index(game_mode),
                pollution_target=pollution_target,
                ui_type=UI_TYPE_CLASSES.index(ui_type),
                uncertain=uncertain,
                group_id=group_id,
            )
        )

    if missing_files:
        print(f"[warn] 有 {len(missing_files)} 条标注没找到图片，已跳过。示例: {missing_files[:5]}")
    if invalid_modes:
        print(f"[warn] 有 {len(invalid_modes)} 条标注缺少有效 game_mode，已跳过。示例: {invalid_modes[:5]}")
    return samples


def split_samples(
    samples: list[Sample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    split_by_group: bool,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    rng = random.Random(seed)

    if split_by_group:
        grouped: dict[str, list[Sample]] = defaultdict(list)
        for s in samples:
            grouped[s.group_id].append(s)
        group_ids = list(grouped.keys())
        rng.shuffle(group_ids)

        total = len(samples)
        target_test = max(1, int(round(total * test_ratio))) if test_ratio > 0 else 0
        target_val = max(1, int(round(total * val_ratio))) if val_ratio > 0 else 0

        train_groups, val_groups, test_groups = [], [], []
        val_count = test_count = 0
        for gid in group_ids:
            group_size = len(grouped[gid])
            if test_count < target_test:
                test_groups.append(gid)
                test_count += group_size
            elif val_count < target_val:
                val_groups.append(gid)
                val_count += group_size
            else:
                train_groups.append(gid)

        train = [s for gid in train_groups for s in grouped[gid]]
        val = [s for gid in val_groups for s in grouped[gid]]
        test = [s for gid in test_groups for s in grouped[gid]]
    else:
        items = samples[:]
        rng.shuffle(items)
        n = len(items)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        test = items[:n_test]
        val = items[n_test:n_test + n_val]
        train = items[n_test + n_val:]

    if not train:
        raise ValueError("训练集为空，请减少 val/test 比例。")
    if not val:
        raise ValueError("验证集为空，请减少 val 比例，或者增加样本。")
    return train, val, test


class AverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


@torch.no_grad()
def multilabel_micro_f1(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).int()
    t = targets.int()
    tp = int(((preds == 1) & (t == 1)).sum().item())
    fp = int(((preds == 1) & (t == 0)).sum().item())
    fn = int(((preds == 0) & (t == 1)).sum().item())
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 1.0


@torch.no_grad()
def binary_subset_accuracy(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).int()
    t = targets.int()
    exact = (preds == t).all(dim=1).float().mean().item()
    return float(exact)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    bce_loss: nn.Module,
    ce_loss: nn.Module,
) -> dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    game_correct = 0
    ui_correct = 0
    total = 0
    all_pollution_logits = []
    all_pollution_targets = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        game_mode = batch["game_mode"].to(device, non_blocking=True)
        pollution = batch["pollution"].to(device, non_blocking=True)
        ui_type = batch["ui_type"].to(device, non_blocking=True)

        outputs = model(images)
        loss_game = ce_loss(outputs["game_mode"], game_mode)
        loss_pollution = bce_loss(outputs["pollution"], pollution)
        loss_ui = ce_loss(outputs["ui_type"], ui_type)
        loss = loss_game + loss_pollution + loss_ui
        loss_meter.update(float(loss.item()), images.size(0))

        game_pred = outputs["game_mode"].argmax(dim=1)
        ui_pred = outputs["ui_type"].argmax(dim=1)
        game_correct += int((game_pred == game_mode).sum().item())
        ui_correct += int((ui_pred == ui_type).sum().item())
        total += images.size(0)

        all_pollution_logits.append(outputs["pollution"].detach().cpu())
        all_pollution_targets.append(pollution.detach().cpu())

    pollution_logits = torch.cat(all_pollution_logits, dim=0)
    pollution_targets = torch.cat(all_pollution_targets, dim=0)
    return {
        "loss": loss_meter.avg,
        "game_mode_acc": game_correct / max(total, 1),
        "ui_type_acc": ui_correct / max(total, 1),
        "pollution_micro_f1": multilabel_micro_f1(pollution_logits, pollution_targets),
        "pollution_subset_acc": binary_subset_accuracy(pollution_logits, pollution_targets),
    }


def print_dataset_stats(name: str, samples: list[Sample]) -> None:
    print(f"\n[{name}] 样本数: {len(samples)}")
    game_counter = Counter(GAME_MODE_CLASSES[s.game_mode] for s in samples)
    ui_counter = Counter(UI_TYPE_CLASSES[s.ui_type] for s in samples)
    pollution_counter = Counter()
    for s in samples:
        for label, value in zip(POLLUTION_LABELS, s.pollution_target):
            if value > 0:
                pollution_counter[label] += 1

    print("  game_mode:", dict(game_counter))
    print("  ui_type:", dict(ui_counter))
    print("  pollution_types:", dict(pollution_counter))


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
        "args": vars(args),
        "class_names": {
            "game_mode": GAME_MODE_CLASSES,
            "pollution_types": POLLUTION_LABELS,
            "ui_type": UI_TYPE_CLASSES,
        },
        "input_spec": {
            "mode": "grayscale",
            "size": [320, 640],  # H, W
            "note": "灰度图，不是二值化图",
        },
    }
    torch.save(ckpt, output_dir / "best.pt")
    with (output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


@torch.no_grad()
def save_test_predictions(model: nn.Module, loader: DataLoader, device: torch.device, output_dir: Path) -> None:
    if len(loader.dataset) == 0:
        return
    model.eval()
    rows = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        outputs = model(images)
        game_pred = outputs["game_mode"].argmax(dim=1).cpu().tolist()
        ui_pred = outputs["ui_type"].argmax(dim=1).cpu().tolist()
        pollution_prob = torch.sigmoid(outputs["pollution"]).cpu().tolist()
        for rel_path, gm, ui, probs in zip(batch["rel_path"], game_pred, ui_pred, pollution_prob):
            rows.append(
                {
                    "image": rel_path,
                    "pred_game_mode": GAME_MODE_CLASSES[gm],
                    "pred_ui_type": UI_TYPE_CLASSES[ui],
                    "pred_pollution_types": [
                        label for label, p in zip(POLLUTION_LABELS, probs) if p >= 0.5
                    ],
                    "pred_pollution_probs": {label: round(float(p), 4) for label, p in zip(POLLUTION_LABELS, probs)},
                }
            )
    with (output_dir / "test_predictions.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def build_transforms(train: bool) -> transforms.Compose:
    ops = [
        transforms.Resize((320, 640)),  # H=320, W=640
    ]
    if train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.15, contrast=0.15)],
                    p=0.4,
                ),
            ]
        )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    return transforms.Compose(ops)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 Minecraft 数据清洗器（多头分类）")
    parser.add_argument("--frames", type=Path, default=Path("frames"), help="图片目录，默认 ./frames")
    parser.add_argument("--annotations", type=Path, default=Path("annotations.json"), help="标注 JSON 文件")
    parser.add_argument("--output", type=Path, default=Path("runs/cleaner_baseline"), help="输出目录")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-uncertain", action="store_true", help="是否把 uncertain=1 的样本也用于训练")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="不使用 ImageNet 预训练 ResNet18（默认使用预训练并改成单通道）",
    )
    parser.add_argument(
        "--no-group-split",
        action="store_true",
        help="关闭按组切分；默认尽量按同一视频/序列分组切分，减少相邻帧泄漏",
    )
    parser.add_argument(
        "--group-regex",
        type=str,
        default=None,
        help="从文件名提取组 ID 的正则；若有捕获组，取第 1 组。例如 '^(.*)_\\d+$'",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device()

    print(f"[info] device = {device}")
    print(f"[info] frames = {args.frames}")
    print(f"[info] annotations = {args.annotations}")

    samples = parse_annotation_samples(
        frames_dir=args.frames,
        annotation_path=args.annotations,
        include_uncertain=args.include_uncertain,
        group_regex=args.group_regex,
    )
    if len(samples) < 20:
        raise ValueError(f"可用样本太少：{len(samples)}。至少先准备几十张再训练。")

    train_samples, val_samples, test_samples = split_samples(
        samples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        split_by_group=not args.no_group_split,
    )

    print_dataset_stats("train", train_samples)
    print_dataset_stats("val", val_samples)
    print_dataset_stats("test", test_samples)

    train_ds = FrameAnnotationDataset(train_samples, build_transforms(train=True))
    val_ds = FrameAnnotationDataset(val_samples, build_transforms(train=False))
    test_ds = FrameAnnotationDataset(test_samples, build_transforms(train=False))

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = MultiTaskCleanerNet(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_score = -math.inf
    args.output.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_meter = AverageMeter()

        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            game_mode = batch["game_mode"].to(device, non_blocking=True)
            pollution = batch["pollution"].to(device, non_blocking=True)
            ui_type = batch["ui_type"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
                loss_game = ce_loss(outputs["game_mode"], game_mode)
                loss_pollution = bce_loss(outputs["pollution"], pollution)
                loss_ui = ce_loss(outputs["ui_type"], ui_type)
                loss = loss_game + loss_pollution + loss_ui

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_meter.update(float(loss.item()), images.size(0))

        val_metrics = evaluate(model, val_loader, device, bce_loss, ce_loss)
        score = val_metrics["game_mode_acc"] + val_metrics["ui_type_acc"] + val_metrics["pollution_micro_f1"]

        print(
            f"epoch {epoch:02d} | train_loss={loss_meter.avg:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_game_acc={val_metrics['game_mode_acc']:.4f} | "
            f"val_ui_acc={val_metrics['ui_type_acc']:.4f} | "
            f"val_pollution_f1={val_metrics['pollution_micro_f1']:.4f} | "
            f"val_pollution_subset={val_metrics['pollution_subset_acc']:.4f}"
        )

        if score > best_score:
            best_score = score
            save_checkpoint(args.output, model, optimizer, epoch, val_metrics, args)
            print(f"[info] 已保存更优模型到 {args.output / 'best.pt'}")

    best_ckpt = torch.load(args.output / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])

    test_metrics = evaluate(model, test_loader, device, bce_loss, ce_loss)
    print("\n[test]")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    with (args.output / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)
    save_test_predictions(model, test_loader, device, args.output)

    summary = {
        "device": str(device),
        "total_samples": len(samples),
        "train": len(train_samples),
        "val": len(val_samples),
        "test": len(test_samples),
        "best_val_score": best_score,
        "test_metrics": test_metrics,
        "input": {
            "grayscale": True,
            "size": [320, 640],
            "binarized": False,
        },
    }
    with (args.output / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[done] 训练完成，结果保存在: {args.output}")
    print("主要文件:")
    print(f"  - {args.output / 'best.pt'}")
    print(f"  - {args.output / 'best_metrics.json'}")
    print(f"  - {args.output / 'test_metrics.json'}")
    print(f"  - {args.output / 'test_predictions.json'}")


if __name__ == "__main__":
    main()
