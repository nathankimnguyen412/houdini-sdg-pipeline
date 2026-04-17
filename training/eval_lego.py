"""
Evaluate trained Lego detector on real-world photos using COCO metrics.

Usage:
  python eval_lego.py

Runs evaluation on both controlled and diverse sets and prints AP metrics.
"""

import os
import json
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# ============================================================
# PATHS
# ============================================================
EVAL_ROOT    = "/path/to/your/eval/directory"
MODEL_PATH   = "/path/to/your/model_final.pth"
OUTPUT_DIR   = "/path/to/your/output/eval_results"
CONTROLLED_ANNOTATIONS = "controlled_annotations.json"
DIVERSE_ANNOTATIONS    = "diverse_annotations.json"

# ============================================================
# EVALUATION ANNOTATIONS
# Ground truth annotations are in dataset/eval_annotations/1class/ (v1, v2)
# and dataset/eval_annotations/7class/ (v3, v4). Photos are in dataset/eval_photos/.
# ============================================================
EVAL_SETS = {
    "lego_real_controlled": {
        "annotations": os.path.join(EVAL_ROOT, "controlled_512", CONTROLLED_ANNOTATIONS),
        "images": os.path.join(EVAL_ROOT, "controlled_512"),
    },
    "lego_real_diverse": {
        "annotations": os.path.join(EVAL_ROOT, "diverse_512", DIVERSE_ANNOTATIONS),
        "images": os.path.join(EVAL_ROOT, "diverse_512"),
    },
}


# ============================================================
# SETUP
# ============================================================
def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LEGO BRICK DETECTOR — COCO EVALUATION")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"GPU:   {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # Register eval datasets
    for name, paths in EVAL_SETS.items():
        print(f"\nRegistering {name}:")
        print(f"  Annotations: {paths['annotations']}")
        print(f"  Images:      {paths['images']}")
        register_coco_instances(name, {}, paths["annotations"], paths["images"])

    cfg = setup_cfg()

    all_results = {}

    for dataset_name in EVAL_SETS:
        print(f"\n{'=' * 60}")
        print(f"EVALUATING: {dataset_name}")
        print(f"{'=' * 60}")

        eval_output = os.path.join(OUTPUT_DIR, dataset_name)
        os.makedirs(eval_output, exist_ok=True)

        evaluator = COCOEvaluator(
            dataset_name,
            tasks=("bbox", "segm"),
            output_dir=eval_output,
        )
        val_loader = build_detection_test_loader(cfg, dataset_name)
        results = inference_on_dataset(
            DefaultPredictor(cfg).model,
            val_loader,
            evaluator,
        )

        all_results[dataset_name] = results

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'Controlled':>12} {'Diverse':>12}")
    print("-" * 54)

    controlled = all_results.get("lego_real_controlled", {})
    diverse = all_results.get("lego_real_diverse", {})

    metrics = [
        ("AP@50 (bbox)", "bbox", "AP50"),
        ("AP@50:95 (bbox)", "bbox", "AP"),
        ("AP@50 (segm)", "segm", "AP50"),
        ("AP@50:95 (segm)", "segm", "AP"),
    ]

    for label, task, key in metrics:
        c_val = controlled.get(task, {}).get(key, -1)
        d_val = diverse.get(task, {}).get(key, -1)
        c_str = f"{c_val:.1f}" if c_val >= 0 else "N/A"
        d_str = f"{d_val:.1f}" if d_val >= 0 else "N/A"
        print(f"{label:<30} {c_str:>12} {d_str:>12}")

    print("=" * 60)

    # Save summary to file
    summary_path = os.path.join(OUTPUT_DIR, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to: {summary_path}")
