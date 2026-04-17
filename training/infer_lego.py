"""
Run inference with trained Lego detector on images.

Usage:
  # Run on 10 random synthetic images (sanity check)
  python infer_lego.py --synthetic 10

  # Run on a folder of real photos
  python infer_lego.py --input /path/to/photos/

  # Run on a single image
  python infer_lego.py --input /path/to/photo.jpg
"""

import argparse
import glob
import os
import random

import cv2
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# ============================================================
# PATHS
# ============================================================
DATASET_ROOT = "/path/to/your/dataset"
ANNOTATIONS  = os.path.join(DATASET_ROOT, "annotations.json")
IMAGES_DIR   = os.path.join(DATASET_ROOT, "images")
MODEL_PATH   = "/path/to/your/model_final.pth"
OUTPUT_DIR   = "/path/to/your/output/predictions"


# ============================================================
# SETUP
# ============================================================
def setup_cfg(confidence_threshold=0.5):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # Update to match your trained model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def run_inference(image_paths, output_dir, confidence=0.5):
    """Run the trained model on a list of images and save predictions."""
    os.makedirs(output_dir, exist_ok=True)

    # Register dataset so metadata (class names) is available
    if "lego_synth_train" not in MetadataCatalog.list():
        register_coco_instances("lego_synth_train", {}, ANNOTATIONS, IMAGES_DIR)
    metadata = MetadataCatalog.get("lego_synth_train")

    cfg = setup_cfg(confidence_threshold=confidence)
    predictor = DefaultPredictor(cfg)

    print(f"Running inference on {len(image_paths)} images (confidence >= {confidence})...\n")

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"  WARNING: Could not read {img_path}, skipping")
            continue

        # Run model
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        num_detections = len(instances)

        # Draw predictions
        vis = Visualizer(
            img[:, :, ::-1],
            metadata=metadata,
            scale=1.0,
            instance_mode=ColorMode.IMAGE_BW,  # Dims non-detected regions
        )
        out = vis.draw_instance_predictions(instances)

        # Save
        basename = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_dir, f"pred_{basename}.png")
        cv2.imwrite(out_path, out.get_image()[:, :, ::-1])

        # Print per-detection details
        if num_detections > 0:
            scores = instances.scores.tolist()
            score_str = ", ".join([f"{s:.2f}" for s in scores])
            print(f"  {os.path.basename(img_path)}: {num_detections} detections (scores: {score_str})")
        else:
            print(f"  {os.path.basename(img_path)}: NO detections")

    print(f"\nPredictions saved to {output_dir}/")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Lego detector inference")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to an image file or folder of images")
    parser.add_argument("--synthetic", type=int, default=None,
                        help="Run on N random synthetic images (sanity check)")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Detection confidence threshold (default: 0.5)")
    args = parser.parse_args()

    if args.synthetic:
        # Grab N random images from the synthetic dataset
        all_images = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.png")))
        image_paths = random.sample(all_images, min(args.synthetic, len(all_images)))
        out_dir = os.path.join(OUTPUT_DIR, "synthetic")
        print(f"Sanity check: running on {len(image_paths)} random synthetic images\n")

    elif args.input:
        if os.path.isfile(args.input):
            image_paths = [args.input]
        elif os.path.isdir(args.input):
            image_paths = sorted(
                glob.glob(os.path.join(args.input, "*.png")) +
                glob.glob(os.path.join(args.input, "*.jpg")) +
                glob.glob(os.path.join(args.input, "*.jpeg"))
            )
        else:
            print(f"ERROR: {args.input} is not a valid file or directory")
            exit(1)
        out_dir = os.path.join(OUTPUT_DIR, "custom")
        print(f"Running on {len(image_paths)} images from {args.input}\n")

    else:
        print("Provide either --synthetic N or --input /path/to/images")
        print("Examples:")
        print("  python infer_lego.py --synthetic 10")
        print("  python infer_lego.py --input /path/to/real_photos/")
        exit(0)

    run_inference(image_paths, out_dir, confidence=args.confidence)
