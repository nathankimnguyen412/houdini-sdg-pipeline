"""
Train Mask R-CNN on synthetic Lego brick dataset.
Usage: python train_lego.py

Expects:
  - annotations.json and images/ folder at DATASET_ROOT
  - Output saved to OUTPUT_DIR
"""

import os
import random
import cv2
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import Visualizer

# ============================================================
# PATHS — Update these to match your setup
# ============================================================
DATASET_ROOT = "/path/to/your/dataset"
ANNOTATIONS  = os.path.join(DATASET_ROOT, "annotations.json")
IMAGES_DIR   = os.path.join(DATASET_ROOT, "images")
OUTPUT_DIR   = "/path/to/your/output/directory"

# ============================================================
# REGISTER DATASET
# ============================================================
register_coco_instances(
    "lego_synth_train",
    {},
    ANNOTATIONS,
    IMAGES_DIR,
)

# ============================================================
# VERIFY DATASET (visual spot-check before training)
# ============================================================
def verify_dataset(num_samples=5):
    """Save a few annotated images so you can visually confirm
    that bounding boxes and masks look correct."""
    verify_dir = os.path.join(OUTPUT_DIR, "verify")
    os.makedirs(verify_dir, exist_ok=True)

    dataset_dicts = DatasetCatalog.get("lego_synth_train")
    metadata = MetadataCatalog.get("lego_synth_train")

    samples = random.sample(dataset_dicts, min(num_samples, len(dataset_dicts)))
    for d in samples:
        img = cv2.imread(d["file_name"])
        if img is None:
            print(f"WARNING: Could not read {d['file_name']}")
            continue
        vis = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        out = vis.draw_dataset_dict(d)
        out_path = os.path.join(verify_dir, f"verify_{d['image_id']}.png")
        cv2.imwrite(out_path, out.get_image()[:, :, ::-1])
        print(f"  Saved: {out_path}")

    print(f"\nVerification images saved to {verify_dir}/")
    print("Check these before training to confirm annotations look correct.\n")


# ============================================================
# TRAINING CONFIG
# ============================================================
def setup_cfg():
    cfg = get_cfg()

    # Base model: Mask R-CNN with ResNet-50-FPN backbone
    # Pretrained on COCO — gives us a strong starting point
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )

    # Dataset
    cfg.DATASETS.TRAIN = ("lego_synth_train",)
    cfg.DATASETS.TEST = ()  # No test set during training

    # Dataloader
    cfg.DATALOADER.NUM_WORKERS = 2

    # Solver (training hyperparameters)
    cfg.SOLVER.IMS_PER_BATCH = 2        # 2 images per batch (fits in 24GB easily)
    cfg.SOLVER.BASE_LR = 0.0025         # Learning rate
    cfg.SOLVER.MAX_ITER = 25000          # ~7 epochs for 7,000 images
    cfg.SOLVER.STEPS = [17500, 22500]               # Decay LR by 10x at these iterations
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # Save checkpoint every 1000 iters

    # Model head
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # 7 Lego part classes

    # Output
    cfg.OUTPUT_DIR = OUTPUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LEGO BRICK DETECTOR — TRAINING")
    print("=" * 60)
    print(f"Dataset:     {ANNOTATIONS}")
    print(f"Images:      {IMAGES_DIR}")
    print(f"Output:      {OUTPUT_DIR}")
    print(f"GPU:         {torch.cuda.get_device_name(0)}")
    print(f"CUDA:        {torch.version.cuda}")
    print("=" * 60)

    # Step 1: Verify dataset loads correctly
    print("\n[1/2] Verifying dataset (saving 5 sample images)...")
    verify_dataset(num_samples=5)

    # Step 2: Train
    print("[2/2] Starting training...")
    cfg = setup_cfg()
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Model saved to: {os.path.join(OUTPUT_DIR, 'model_final.pth')}")
    print("=" * 60)
