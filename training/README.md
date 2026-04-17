# Training Scripts

Scripts for training and evaluating a Mask R-CNN on the synthetic Lego dataset.

## Dependencies

- Python 3.10
- Detectron2 0.6
- PyTorch 2.10.0+cu130
- CUDA-compatible GPU (tested on NVIDIA RTX 3090, 24GB VRAM)

## Workflow

Run in this order:

1. `resize_photos.py` — center-crop and resize real evaluation photos to 512×512
2. `train_lego.py` — fine-tune Mask R-CNN on the synthetic dataset
3. `infer_lego.py` — run inference and save visualizations for qualitative review
4. `eval_lego.py` — compute COCO AP metrics against ground truth annotations

## Script Descriptions

**resize_photos.py** — Preprocesses real-world phone photos for evaluation. Takes high-resolution images, center-crops to square, and resizes to 512×512 to match synthetic training resolution.
```bash
python resize_photos.py --input /path/to/raw/photos --output /path/to/resized/
```

**train_lego.py** — Fine-tunes Mask R-CNN with ResNet-50-FPN backbone from COCO-pretrained weights on the synthetic dataset. Outputs `model_final.pth`.
```bash
python train_lego.py
```

**infer_lego.py** — Runs the trained model on images and saves prediction visualizations. Use `--synthetic` for a sanity check on training data, or `--input` for real photos.
```bash
python infer_lego.py --synthetic 10
python infer_lego.py --input /path/to/photos/
```

**eval_lego.py** — Evaluates the trained model against COCO ground truth annotations and reports AP@50 and AP@50:95 for both bbox and segmentation on the controlled and diverse eval sets.
```bash
python eval_lego.py
```

## Training Configurations

All runs use the same base settings unless noted:

- Architecture: Mask R-CNN with ResNet-50-FPN backbone
- Pretrained weights: COCO (mask_rcnn_R_50_FPN_3x)
- Image resolution: 512×512
- Batch size: 2
- Base learning rate: 0.0025
- ROI heads batch size per image: 128
- Confidence threshold: 0.5
- Hardware: NVIDIA RTX 3090, WSL2 (Ubuntu)

| Parameter | v1 | v2 | v3 | v4 |
|-----------|:--:|:--:|:--:|:--:|
| NUM_CLASSES | 1 | 1 | 7 | 7 |
| MAX_ITER | 5,000 | 5,000 | 15,000 | 25,000 |
| LR_STEPS | none | none | [10500, 13500] | [17500, 22500] |
| Distractors | No | Yes | Yes | Yes |
| Training time | ~10 min | ~10 min | ~29 min | ~48 min |

## Dataset Paths

Update the path constants at the top of each script to match your local setup:
```python
DATASET_ROOT = "/path/to/your/dataset"
MODEL_PATH   = "/path/to/your/model_final.pth"
```

## Notes

- Scripts are configured for the v4 7-class model by default. To reproduce earlier runs, update `NUM_CLASSES`, `MAX_ITER`, and `LR_STEPS` per the table above.
- The Detectron2 environment was set up in WSL2 using conda. Activate with `conda activate detectron2` before running any script.
- Evaluation ground truth annotations were produced using SAM 3 in CVAT with manual correction. See the main README for annotation methodology details.
- Full evaluation results for all four training runs are saved in `assets/results/` as `results_summary_v1.json` through `results_summary_v4.json`.