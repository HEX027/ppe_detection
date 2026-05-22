#!/usr/bin/env python3
"""
train.py — Fine-tune YOLOv8 for PPE Detection
==============================================================
Usage:
    python3 src/train.py --config configs/ppe_dataset.yaml \
                         --epochs 100 --batch 16 --model yolov8m.pt

What this script does
─────────────────────
1. Downloads / verifies the PPE dataset (Roboflow + custom frames)
2. Defines an Albumentations augmentation pipeline covering:
   horizontal flip, mosaic, HSV shifts, scale jitter, cutout
3. Fine-tunes a YOLOv8 model via transfer learning (COCO weights)
4. Evaluates on the validation split: mAP@0.5, Precision, Recall, F1
5. Saves the best weights to models/weights/best.pt
==============================================================
"""

import argparse
import os
import sys
import time
from pathlib import Path

import albumentations as A
import numpy as np
import yaml
from loguru import logger
from ultralytics import YOLO

# ── Ensure project root is on PYTHONPATH ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────
# 1. CLI Arguments
# ─────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPE YOLOv8 Training Script")
    parser.add_argument(
        "--config", type=str,
        default="configs/ppe_dataset.yaml",
        help="Path to the YOLOv8 dataset YAML config"
    )
    parser.add_argument(
        "--model", type=str,
        default="yolov8m.pt",           # COCO-pretrained medium model
        help="YOLOv8 variant: yolov8n/s/m/l/x.pt"
    )
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch",      type=int,   default=16,
                        help="Batch size (reduce if OOM on Jetson)")
    parser.add_argument("--imgsz",      type=int,   default=640)
    parser.add_argument("--workers",    type=int,   default=4)
    parser.add_argument("--device",     type=str,   default="0",
                        help="CUDA device index or 'cpu'")
    parser.add_argument("--project",    type=str,   default="runs/train")
    parser.add_argument("--name",       type=str,   default="ppe_finetune")
    parser.add_argument("--resume",     action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--patience",   type=int,   default=20,
                        help="Early-stopping patience (epochs)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────
# 2. Dataset Download Helper (Roboflow)
# ─────────────────────────────────────────────────────────
def download_roboflow_dataset(
    api_key: str,
    workspace: str,
    project: str,
    version: int,
    output_dir: str = "data/raw"
) -> str:
    """
    Download the Roboflow PPE dataset via the Roboflow Python SDK.

    Returns the path to the exported dataset YAML.

    Prerequisites:
        pip install roboflow
        export ROBOFLOW_API_KEY=your_key
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error("roboflow not installed. Run: pip install roboflow")
        sys.exit(1)

    logger.info(f"Downloading Roboflow dataset: {workspace}/{project} v{version}")
    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    dataset = project_obj.version(version).download(
        "yolov8",
        location=output_dir
    )
    yaml_path = os.path.join(output_dir, "data.yaml")
    logger.success(f"Dataset saved to {output_dir}, YAML at {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────────────────────
# 3. Albumentations Augmentation Pipeline
#    (Used for custom preprocessing / offline augmentation.
#     YOLOv8's built-in augmentations are controlled via
#     hyper-parameter overrides in train() below.)
# ─────────────────────────────────────────────────────────
def build_augmentation_pipeline(image_size: int = 640) -> A.Compose:
    """
    Returns an Albumentations Compose pipeline covering:
      • Horizontal Flip
      • Mosaic  (simulated via RandomResizedCrop + grid paste)
      • HSV color shifts
      • Scale Jitter
      • Cutout (CoarseDropout)
    
    bbox_params enables bounding-box-aware transforms so that
    labels are correctly updated when the image is modified.
    """
    return A.Compose(
        [
            # ── Geometric ─────────────────────────────────
            # Horizontal flip (50 % probability)
            A.HorizontalFlip(p=0.5),

            # Scale jitter — randomly zoom/crop within ±20%
            A.RandomResizedCrop(
                height=image_size,
                width=image_size,
                scale=(0.8, 1.2),
                ratio=(0.75, 1.33),
                p=0.5,
            ),

            # Random rotation ±15 degrees
            A.Rotate(limit=15, border_mode=0, p=0.3),

            # Perspective distortion (simulates different camera angles)
            A.Perspective(scale=(0.05, 0.1), p=0.2),

            # ── Colour / Appearance ───────────────────────
            # HSV colour shifts (hue, saturation, value independently)
            A.HueSaturationValue(
                hue_shift_limit=20,       # ±20 degrees on the hue wheel
                sat_shift_limit=30,       # ±30 saturation
                val_shift_limit=20,       # ±20 brightness
                p=0.5,
            ),

            # Random brightness & contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.4,
            ),

            # Simulate real-world fog / haze (common in industrial env.)
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),

            # Gaussian noise (sensor noise simulation)
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),

            # Motion blur (fast-moving workers)
            A.MotionBlur(blur_limit=5, p=0.15),

            # ── Cutout / Occlusion ────────────────────────
            # CoarseDropout implements the Cutout paper:
            # randomly masks out square regions of the image.
            # max_holes × max_height × max_width approximates
            # masking ~5-15 % of the image area.
            A.CoarseDropout(
                max_holes=8,
                max_height=int(image_size * 0.1),   # 10% of image H
                max_width=int(image_size * 0.1),    # 10% of image W
                min_holes=1,
                fill_value=0,       # fill with black (matches YOLO background)
                p=0.3,
            ),

            # ── Normalisation ─────────────────────────────
            # Standard ImageNet normalisation (YOLO applies its own
            # normalisation internally; this is for offline pipelines)
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                p=1.0,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",           # [x_center, y_center, w, h] normalised
            label_fields=["class_labels"],
            min_area=256,            # drop boxes < 16×16 px after transform
            min_visibility=0.3,      # drop boxes <30% visible after crop
        ),
    )


# ─────────────────────────────────────────────────────────
# 4. Training
# ─────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> YOLO:
    """
    Fine-tune a COCO-pretrained YOLOv8 model on the PPE dataset.

    Key hyper-parameters that implement the required augmentations
    are passed directly to model.train():

      mosaic      — mosaic composition (4-image mosaic, default=1.0)
      flipud      — vertical flip probability
      fliplr      — horizontal flip probability
      hsv_h/s/v   — HSV colour augmentation magnitudes
      scale       — scale jitter magnitude
      copy_paste  — copy-paste augmentation
    """
    logger.info("=" * 60)
    logger.info("  PPE Detection — YOLOv8 Fine-Tuning")
    logger.info("=" * 60)
    logger.info(f"  Base model  : {args.model}")
    logger.info(f"  Dataset     : {args.config}")
    logger.info(f"  Epochs      : {args.epochs}")
    logger.info(f"  Batch size  : {args.batch}")
    logger.info(f"  Image size  : {args.imgsz}")
    logger.info(f"  Device      : {args.device}")
    logger.info("=" * 60)

    # Load pre-trained COCO weights (transfer learning baseline)
    model = YOLO(args.model)

    # ── Start training ────────────────────────────────────
    results = model.train(
        # Dataset
        data=args.config,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,

        # ── Augmentation hyper-parameters ─────────────────
        # Mosaic composition (probability: 1.0 = always enabled
        # for the first 90% of training, then disabled)
        mosaic=1.0,

        # Horizontal flip (mirrors the Albumentations pipeline)
        fliplr=0.5,

        # HSV colour shifts
        hsv_h=0.015,      # hue shift fraction of 180°
        hsv_s=0.7,        # saturation gain
        hsv_v=0.4,        # value (brightness) gain

        # Scale jitter (±50% of image size)
        scale=0.5,

        # Copy-paste augmentation (segment-based, helps with occlusion)
        copy_paste=0.1,

        # MixUp probability
        mixup=0.1,

        # Perspective distortion
        perspective=0.0005,

        # Degrees of random rotation
        degrees=10.0,

        # ── Transfer learning / regularisation ────────────
        # Freeze the backbone for first N epochs, then unfreeze
        freeze=10,          # freeze first 10 layers

        # Warmup + cosine LR schedule
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        lr0=0.01,           # initial learning rate
        lrf=0.01,           # final LR fraction (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,

        # Early stopping
        patience=args.patience,

        # Output
        project=args.project,
        name=args.name,
        exist_ok=True,
        resume=args.resume,

        # Logging
        verbose=True,
        plots=True,         # generate training curve PNG files
        save=True,
        save_period=10,     # checkpoint every 10 epochs
    )

    logger.success(f"Training complete. Results saved to: {results.save_dir}")
    return model


# ─────────────────────────────────────────────────────────
# 5. Evaluation — mAP@0.5, Precision, Recall, F1
# ─────────────────────────────────────────────────────────
def evaluate(model: YOLO, dataset_yaml: str, image_size: int, device: str) -> dict:
    """
    Run validation on the val split and print:
      • mAP@0.5
      • mAP@0.5:0.95
      • Precision (mean across classes)
      • Recall    (mean across classes)
      • F1-score  (harmonic mean of P and R)

    Target thresholds:
      mAP@0.5  ≥ 0.80
      Precision ≥ 0.85
      Recall    ≥ 0.80
      F1-score  ≥ 0.82
    """
    logger.info("Running evaluation on validation split …")

    metrics = model.val(
        data=dataset_yaml,
        imgsz=image_size,
        device=device,
        verbose=True,
        plots=True,
    )

    # ── Extract per-metric scalars ─────────────────────────
    mp      = float(metrics.box.mp)          # mean Precision
    mr      = float(metrics.box.mr)          # mean Recall
    map50   = float(metrics.box.map50)       # mAP @ IoU=0.50
    map5095 = float(metrics.box.map)         # mAP @ IoU=0.50:0.95

    # F1-score: harmonic mean of Precision and Recall
    f1 = (2 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0.0

    # ── Per-class breakdown ────────────────────────────────
    class_names = metrics.names  # {0: 'helmet', 1: 'vest', 2: 'gloves'}
    per_class_ap = metrics.box.ap50  # shape [num_classes]

    logger.info("\n" + "=" * 60)
    logger.info("  EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  mAP @ 0.50        : {map50:.4f}  (target ≥ 0.80)")
    logger.info(f"  mAP @ 0.50:0.95   : {map5095:.4f}")
    logger.info(f"  Precision (mean)  : {mp:.4f}  (target ≥ 0.85)")
    logger.info(f"  Recall    (mean)  : {mr:.4f}  (target ≥ 0.80)")
    logger.info(f"  F1-score  (mean)  : {f1:.4f}  (target ≥ 0.82)")
    logger.info("-" * 60)
    logger.info("  Per-class AP@0.50:")
    for idx, ap in enumerate(per_class_ap):
        cname = class_names.get(idx, f"class_{idx}")
        logger.info(f"    [{idx}] {cname:<12}: {float(ap):.4f}")
    logger.info("=" * 60)

    # ── Target threshold checks ───────────────────────────
    thresholds = {
        "mAP@0.50":    (map50,  0.80),
        "Precision":   (mp,     0.85),
        "Recall":      (mr,     0.80),
        "F1-score":    (f1,     0.82),
    }
    all_passed = True
    for metric_name, (value, target) in thresholds.items():
        status = "✅ PASS" if value >= target else "❌ FAIL"
        logger.info(f"  {metric_name:<15}: {value:.4f} {status} (≥{target})")
        if value < target:
            all_passed = False

    if all_passed:
        logger.success("All target thresholds met — model is deployment-ready.")
    else:
        logger.warning("Some thresholds not met — consider more training or data.")

    return {
        "map50":   map50,
        "map5095": map5095,
        "precision": mp,
        "recall":    mr,
        "f1":        f1,
    }


# ─────────────────────────────────────────────────────────
# 6. Entry Point
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    # ── (Optional) Roboflow dataset download ──────────────
    # Uncomment and fill in credentials to auto-download:
    #
    # api_key   = os.environ.get("ROBOFLOW_API_KEY", "")
    # yaml_path = download_roboflow_dataset(
    #     api_key   = api_key,
    #     workspace = "your-workspace",
    #     project   = "ppe-detection",
    #     version   = 3,
    #     output_dir= "data/raw/roboflow"
    # )
    # args.config = yaml_path

    # ── Train ─────────────────────────────────────────────
    start = time.time()
    trained_model = train(args)
    elapsed = time.time() - start
    logger.info(f"Training time: {elapsed / 3600:.2f} hours")

    # ── Evaluate ──────────────────────────────────────────
    eval_results = evaluate(
        model=trained_model,
        dataset_yaml=args.config,
        image_size=args.imgsz,
        device=args.device,
    )

    # ── Copy best weights to canonical location ───────────
    best_weights_src = Path(args.project) / args.name / "weights" / "best.pt"
    best_weights_dst = PROJECT_ROOT / "models" / "weights" / "best.pt"
    best_weights_dst.parent.mkdir(parents=True, exist_ok=True)

    if best_weights_src.exists():
        import shutil
        shutil.copy(best_weights_src, best_weights_dst)
        logger.success(f"Best weights saved to: {best_weights_dst}")
    else:
        logger.warning(f"Best weights not found at: {best_weights_src}")
