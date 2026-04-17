# Houdini Synthetic Data Generation Pipeline

> Procedural synthetic data generation for instance segmentation and object detection, built in SideFX Houdini.

![Pipeline Banner](assets/banner.png)

---

## Overview

This project is an end-to-end procedural synthetic data generation pipeline for Lego brick instance segmentation and object detection. It was built in SideFX Houdini across SOPs, Solaris, and TOPs/PDG and is designed to produce COCO-format datasets without any manually annotated fine-tuning data. The pipeline renders scenes using Karma XPU and extracts COCO-format annotations automatically with no manual labeling. Domain randomization across 32 axes drives sim-to-real transfer, covering lighting, materials, camera pose, object placement, occlusion, background texture, and more. A Mask R-CNN with ResNet-50-FPN backbone was initialized from COCO pretraining and fine-tuned exclusively on pipeline output. The model was then evaluated against two real-world photograph datasets (Controlled, n=30; Diverse, n=60), with evaluation ground truth produced via SAM 3-assisted annotation with manual correction in CVAT, achieving AP@50 of 94.0% on bbox and 95.1% on segmentation under diverse, uncontrolled conditions in the single-class baseline. The project follows a full iterative research loop: failure mode analysis on real-world evaluation results drove targeted pipeline interventions including distractor objects, occlusion augmentation, and 13 additional background textures, followed by scale-up to 7,000 images across 7 geometrically distinct part classes.

---

## Assets

### Lego parts

Source geometry for all training scenes comes from [LDraw](https://www.ldraw.org), an open standard for Lego CAD models with a community-maintained library of over 18,000 official parts. LDraw `.dat` files define precise geometry for each part, which are imported into Houdini SOPs and used as the objects to be detected and segmented.

Seven geometrically distinct parts were selected to cover a range of shapes, sizes, and visual complexity:

<table>
  <tr>
    <td align="center"><img src="assets/parts/1x1_plate.PNG" width="120"/><br/>1x1 plate</td>
    <td align="center"><img src="assets/parts/2x4_plate.PNG" width="120"/><br/>2x4 plate</td>
    <td align="center"><img src="assets/parts/1x3_plate.PNG" width="120"/><br/>1x3 plate</td>
    <td align="center"><img src="assets/parts/1x2_plate_with_2_U_Clips.PNG" width="120"/><br/>1x2 plate with 2 U-clips</td>
  </tr>
  <tr>
    <td align="center"><img src="assets/parts/2x2_brick.PNG" width="120"/><br/>2x2 brick</td>
    <td align="center"><img src="assets/parts/1x1_brick_stud_on_side.PNG" width="120"/><br/>1x1 brick with stud on side</td>
    <td align="center"><img src="assets/parts/1x1_round_brick.PNG" width="120"/><br/>1x1 round brick</td>
    <td></td>
  </tr>
</table>

These parts were chosen to create a challenging classification problem: several share similar silhouettes from certain viewpoints (1x3 plate vs 2x4 plate, 1x1 plate vs 1x1 brick with stud on side), directly testing the model's ability to learn fine-grained geometric distinctions from synthetic data alone.

---

### Textures

Ground and backdrop textures were sourced from two libraries:

- [Poly Haven](https://polyhaven.com) - PBR texture sets (albedo, roughness, normal, displacement maps)
- [ambientCG](https://ambientcg.com) - additional PBR texture sets

The texture library covers 30 surface types spanning wood, concrete, tile, fabric, metal, gravel, and coarse outdoor surfaces. 13 additional textures (rough concrete, gravel, coarse stone, and reflective metals) were added in v2 specifically to address false positive failure modes identified in v1 evaluation.

---

### HDRIs

All HDRIs were sourced from [Poly Haven](https://polyhaven.com). 35 HDRIs are used across four categories, weighted toward indoor environments to match the most common real-world use case:

| Category | Count | Description |
|----------|:-----:|-------------|
| Studio / workshop | 8 | Controlled, clean lighting with varied directionality |
| Living spaces | 7 | Kitchen, living room, dining room, bedroom (varied window size and direction) |
| Commercial / other indoor | 5 | Office, warehouse, museum (unusual color casts and lighting patterns) |
| Sunny / clear sky | 4 | Strong directional shadows, blue fill |
| Overcast / cloudy | 4 | Soft diffuse light, muted shadows |
| Golden hour / sunset | 2 | Warm dramatic color cast, extreme white balance |
| Night / dim | 2 | Low light, artificial light sources |
| Mixed lighting | 2 | Multiple color temperatures simultaneously |
| High contrast | 1 | Very bright and very dark regions in the same scene |
| **Total** | **35** | |

HDRI selection was informed by Tremblay et al. (2018), which identifies lighting variety as the single highest-impact domain randomization axis for sim-to-real transfer. The distribution skews indoor (57%) to match the most common conditions in the diverse evaluation set.

One known distribution gap: the current HDRI library skews toward higher-exposure environments, resulting in fewer truly dark or low-light training scenes. The night/dim category (2 HDRIs) is underrepresented relative to real-world conditions.

---

## Pipeline Architecture

![Pipeline Architecture](assets/pipeline_architecture.svg)

---

## Technical Highlights

- **Renderer:** Karma XPU (Solaris)
- **Scene Assembly:** Houdini SOPs (geometry, object placement, intersection avoidance)
- **Task Orchestration:** Houdini TOPs/PDG (batch generation, wedging, parallel task execution)
- **Output Format:** COCO JSON (bounding boxes + instance segmentation masks)
- **Domain Randomization Axes:** 32
- **Dataset Scale:** 1,000 images (v1 and v2) → 3,000 images (v3) → 7,000 images (v4)
- **Classes:** 1 class (v1 and v2) → 7 fine-grained part classes (v3 and v4)
- **Model:** Detectron2 Mask R-CNN with ResNet-50-FPN backbone (COCO pretrained, fine-tuned on synthetic data)
- **Eval Ground Truth:** SAM 3-assisted annotation with manual correction in CVAT
- **Evaluation:** COCO metrics via `eval_lego.py` (AP@50, AP@50:95 for bbox and segmentation)
- **Eval Sets:** Controlled (n=30) and Diverse (n=60) real-world photographs
- **Best single-class diverse AP@50:** 95.1% segmentation, 94.0% bbox (v1)
- **Best 7-class diverse AP@50:** 79.4% segmentation and bbox (v4)
- **Best 7-class controlled AP@50:** 90.6% segmentation and bbox (v4)
- **Largest per-class scaling gain:** 1x1_brick_stud_on_side +38.1 AP@50:95 controlled (v3 → v4)

---

## Domain Randomization

The pipeline randomizes across 32 axes grouped into 7 categories.

### Known dataset limitations

Two geometric placement limitations exist in the current datasets and are documented here for transparency.

**Brick-to-brick intersections:** A small percentage of images contain Lego bricks that intersect rather than merely touching or occluding each other. This occurred as a deliberate tradeoff. The minimum separation distance parameter was reduced to increase the probability of partial occlusion between bricks, which was identified as an important augmentation for improving model robustness. Reducing separation distance increased occlusion variety but also introduced a small number of physically implausible intersecting placements.

**Brick-to-ground overlap:** A small percentage of images contain bricks that exhibit slight overlap with the ground plane. This is caused by inconsistent bounding box dimensions across different brick orientations. Bricks with studs facing upward have different effective heights than bricks with flat faces or studs facing sideways, making a single ground clearance threshold difficult to calibrate across all 7 part types and all placement orientations.

Both limitations are minor relative to the overall dataset quality and do not appear to have significantly affected model performance, as evidenced by the AP scores achieved on real-world evaluation. They are noted here for reproducibility and to guide future pipeline improvements.

![Domain Randomization Grid](assets/domain_randomization_grid.png)

---

## Training & Results

The core question this pipeline answers is whether a model fine-tuned exclusively on procedurally generated synthetic data can detect and classify real Lego bricks in uncontrolled environments. The findings reported here are not claimed as novel contributions to the synthetic data literature. Rather, they serve two purposes: validating that this pipeline independently reproduces results consistent with established domain randomization research (Tremblay et al. 2018, Prakash et al. 2019), and demonstrating that a single-developer Houdini-native pipeline built from first principles can achieve sim-to-real transfer on par with the methodologies described in those foundational papers. To answer the core question, four training runs were conducted using Mask R-CNN with a ResNet-50-FPN backbone fine-tuned from COCO-pretrained weights, on an NVIDIA RTX 3090 in WSL2 using Detectron2 0.6 with PyTorch 2.10.0+cu130. Evaluation ground truth was produced via SAM 3-assisted annotation with manual correction in CVAT.

### Class legend

The following class IDs appear as labels in all prediction visualization images:

| ID | Class |
|----|-------|
| 0 | 1x1_plate |
| 1 | 2x4_plate |
| 2 | 1x3_plate |
| 3 | 1x2_plate_with_2_U_Clips |
| 4 | 2x2_brick |
| 5 | 1x1_brick_stud_on_side |
| 6 | 1x1_round_brick |

### Evaluation sets

| Set | Number of Images | Conditions |
|-----|:--------:|------------|
| Controlled | 30 | Single surface, consistent overhead lighting, ~45° camera angle, 1–3 bricks per image |
| Diverse | 60 | Multiple surfaces (kitchen counter, carpet, outdoor concrete, gravel, grass, stainless steel), varied lighting, angles, distances, and background clutter |

---

### Overall results across all four training runs

| Metric | v1 (1-class, 1000 images) | v2 (1-class, 1000 images) | v3 (7-class, 3000 images) | v4 (7-class, 7000 images) |
|--------|:---:|:---:|:---:|:---:|
| **Controlled** | | | | |
| AP@50 (bbox) | 99.9 | 100.0 | 86.1 | 90.6 |
| AP@50:95 (bbox) | 86.8 | 88.2 | 77.9 | 85.4 |
| AP@50 (segm) | 97.9 | 100.0 | 86.1 | 90.6 |
| AP@50:95 (segm) | 88.2 | 90.1 | 77.2 | 82.2 |
| **Diverse** | | | | |
| AP@50 (bbox) | 94.0 | 93.8 | 73.2 | 79.4 |
| AP@50:95 (bbox) | 78.7 | 78.0 | 64.9 | 70.5 |
| AP@50 (segm) | 95.1 | 94.1 | 73.2 | 79.4 |
| AP@50:95 (segm) | 78.5 | 78.5 | 61.9 | 66.7 |

| | v1 | v2 | v3 | v4 |
|--|:--:|:--:|:--:|:--:|
| Number of Images | 1,000 | 1,000 | 3,000 | 7,000 |
| Number of Classes | 1 | 1 | 7 | 7 |
| Number of Iterations | 5,000 | 5,000 | 15,000 | 25,000 |
| Training time | ~10 min | ~10 min | ~29 min | ~48 min |
| Final total loss | 0.2602 | 0.2202 | 0.1759 | 0.1818 |
| Distractor Objects | No | Yes | Yes | Yes |

![Inference results placeholder](assets/results/inference_grid.png)

---

### Key findings

- **Synthetic-only training transfers effectively to real photographs.** The v1 baseline (1,000 synthetic images, no distractor objects) was trained in 10 minutes and achieved 94.0 AP@50 on diverse real-world photographs. The pipeline's domain randomization was sufficient to close the sim-to-real gap without any real training data.

- **Distractor objects reduce false positives without hurting overall performance.** The v1 to v2 comparison (adding primitive geometric distractor objects and 13 additional ground textures) showed meaningful qualitative improvement on challenging images with the number of false positives dropping. Overall AP@50 held flat across both controlled and diverse sets, and mask precision (AP@50:95 segm) improved by 1.9 points on controlled, confirming that distractors sharpened mask quality without sacrificing detection rate.

- **Multi-class detection is strictly harder than single-class.** The v2 to v3 transition (1-class to 7-class) caused the largest AP drops in the project (controlled AP@50 fell from 100.0 to 86.1, diverse from 93.8 to 73.2). The model now answers two questions simultaneously (detection and classification), and mAP is averaged across classes of varying geometric difficulty.

- **Dataset scale provides meaningful gains for multi-class detection.** Scaling from 3000 to 7000 images (v3 to v4) improved every metric by 4.5–7.5 points. The largest gains were on strict metrics (AP@50:95), indicating more data improves both detection rates and localization precision. Small object AP on diverse jumped from 29.9% to 60.1%.

- **Dataset size requirements are class-dependent.** Geometrically complex classes (1x1_brick_stud_on_side: +38.1 points controlled, v3 to v4) benefit dramatically from more training data. Geometrically simple, distinctive classes (2x4_plate, 2x2_brick) show diminishing returns at 3000 images. This has direct implications for data budgeting in industrial synthetic data pipelines.

---

### v3 per-class results (7-class, 3,000 images)

**Class distribution (3,000 images, balance ratio 1.08x):**

| Class | Instances | % |
|-------|:---------:|:---:|
| 1x1_plate | 1,222 | 13.7% |
| 2x4_plate | 1,239 | 13.8% |
| 1x3_plate | 1,273 | 14.2% |
| 1x2_plate_with_2_U_Clips | 1,280 | 14.3% |
| 2x2_brick | 1,318 | 14.7% |
| 1x1_brick_stud_on_side | 1,317 | 14.7% |
| 1x1_round_brick | 1,298 | 14.5% |
| **Total** | **8,947** | |

**AP@50:95 (bbox):**

| Class | Controlled | Diverse |
|-------|:---:|:---:|
| 2x2_brick | 96.0 | 69.1 |
| 2x4_plate | 94.5 | 74.8 |
| 1x1_round_brick | 89.7 | 68.4 |
| 1x1_plate | 77.5 | 69.9 |
| 1x2_plate_with_2_U_Clips | 77.3 | 64.3 |
| 1x3_plate | 74.9 | 53.8 |
| 1x1_brick_stud_on_side | 35.6 | 54.2 |

**AP@50:95 (segmentation):**

| Class | Controlled | Diverse |
|-------|:---:|:---:|
| 2x2_brick | 93.9 | 67.5 |
| 2x4_plate | 92.7 | 73.5 |
| 1x1_round_brick | 91.4 | 66.2 |
| 1x1_plate | 80.6 | 65.3 |
| 1x2_plate_with_2_U_Clips | 74.2 | 58.2 |
| 1x3_plate | 71.9 | 50.8 |
| 1x1_brick_stud_on_side | 35.6 | 51.7 |

---

### v4 per-class results (7-class, 7,000 images)

**Class distribution (7,000 images, balance ratio 1.04x):**

| Class | Instances | % |
|-------|:---------:|:---:|
| 1x1_plate | 2,924 | 14.0% |
| 2x4_plate | 2,955 | 14.1% |
| 1x3_plate | 3,020 | 14.5% |
| 1x2_plate_with_2_U_Clips | 2,962 | 14.2% |
| 2x2_brick | 3,033 | 14.5% |
| 1x1_brick_stud_on_side | 3,041 | 14.6% |
| 1x1_round_brick | 2,961 | 14.2% |
| **Total** | **20,896** | |

**AP@50:95 (bbox):**

| Class | Controlled | Diverse | Gap |
|-------|:---:|:---:|:---:|
| 2x2_brick | 98.5 | 64.5 | -34.0 |
| 2x4_plate | 96.2 | 79.0 | -17.2 |
| 1x1_round_brick | 91.1 | 78.0 | -13.1 |
| 1x2_plate_with_2_U_Clips | 88.3 | 75.3 | -13.0 |
| 1x1_plate | 79.9 | 75.7 | -4.2 |
| 1x1_brick_stud_on_side | 73.7 | 65.2 | -8.5 |
| 1x3_plate | 70.2 | 56.1 | -14.1 |

**AP@50:95 (segmentation):**

| Class | Controlled | Diverse | Gap |
|-------|:---:|:---:|:---:|
| 2x4_plate | 93.1 | 75.1 | -18.0 |
| 2x2_brick | 93.1 | 63.5 | -29.6 |
| 1x1_round_brick | 92.1 | 73.4 | -18.7 |
| 1x2_plate_with_2_U_Clips | 82.5 | 64.3 | -18.2 |
| 1x1_plate | 75.0 | 71.4 | -3.6 |
| 1x1_brick_stud_on_side | 72.4 | 65.1 | -7.3 |
| 1x3_plate | 67.4 | 54.0 | -13.4 |

---

### v3 to v4 per-class scaling (AP@50:95 bbox)

| Class | v3 Controlled | v4 Controlled | Delta | v3 Diverse | v4 Diverse | Delta |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| 1x1_brick_stud_on_side | 35.6 | 73.7 | +38.1 | 54.2 | 65.2 | +11.0 |
| 1x2_plate_with_2_U_Clips | 77.3 | 88.3 | +11.0 | 64.3 | 75.3 | +11.0 |
| 1x1_round_brick | 89.7 | 91.1 | +1.4 | 68.4 | 78.0 | +9.6 |
| 1x1_plate | 77.5 | 79.9 | +2.4 | 69.9 | 75.7 | +5.8 |
| 2x4_plate | 94.5 | 96.2 | +1.7 | 74.8 | 79.0 | +4.2 |
| 1x3_plate | 74.9 | 70.2 | -4.7 | 53.8 | 56.1 | +2.3 |
| 2x2_brick | 96.0 | 98.5 | +2.5 | 69.1 | 64.5 | -4.6 |

The gains from dataset scaling are not uniform across classes. Geometrically complex classes benefit most: 1x1_brick_stud_on_side went from essentially broken (35.6) to functional (73.7) with approximately 1,750 additional instances across 4,000 additional images. Classes that were already well-learned at 3000 images (2x4_plate at 94.5, 2x2_brick at 96.0) showed minimal additional gains on controlled (classic diminishing returns for objects with simple, distinctive geometry). The 1x3_plate showed a slight regression on controlled (-4.7) while improving on diverse (+2.3), suggesting the larger dataset introduced more viewpoint ambiguity with the visually similar 2x4_plate on clean backgrounds while still improving real-world generalization.

---

### Inference examples

![Inference on controlled set](assets/results/controlled_predictions.png)
![Inference on diverse set](assets/results/diverse_predictions.png)

Full COCO evaluation results for all four runs are available in [`assets/results/`](assets/results/).

---

## Engineering Decisions

An earlier RGB-based mask approach was prototyped and replaced before the first formal training run.

---

### Why Karma XPU

Karma XPU is Houdini's native USD/Solaris renderer with full GPU acceleration. The choice was straightforward for two reasons.

First, GPU acceleration on the RTX 3090. Karma XPU rendered 512×512 scenes in roughly 8-9 seconds per frame depending on scene complexity, making a 7,000-image dataset generation feasible in under 17 hours on a single consumer GPU. A CPU renderer at equivalent quality settings would have been impractical at this scale.

Second, native integration with the Solaris USD pipeline. All material assignments, lighting rigs, HDRI domes, and camera setups live in the USD stage and are read directly by Karma without any translation layer. Adding a third-party renderer would have required a translation step between the USD scene graph and the renderer's native scene format, adding pipeline complexity.

---

### Training configuration

The training configuration follows established practices from the domain randomization literature and Detectron2's own documented defaults, rather than arbitrary choices.

**Architecture:** Mask R-CNN with ResNet-50-FPN was selected because it is the standard baseline used in the foundational DR papers (Tremblay et al. 2018, Prakash et al. 2019), making results directly comparable to the literature.

**COCO pretraining:** Transfer learning from COCO-pretrained weights is standard practice in synthetic data pipelines. Tremblay et al. (2018) explicitly use ImageNet/COCO pretraining before fine-tuning on synthetic data. The backbone's existing knowledge of edges, textures, and shapes reduces the training data required to achieve strong performance.

**Learning rate (0.0025):** Derived from Detectron2's default Mask R-CNN configuration (BASE_LR: 0.02 at batch size 16) using the linear scaling rule from Goyal et al. (2017): 0.02 × (2/16) = 0.0025.

**Iterations:** Calibrated for approximately 7–10 epochs per run. One epoch equals dataset size divided by batch size = 500 iterations per epoch at 1000 images and 3,500 iterations at 7000 images. Detectron2's default schedules are designed for COCO's 118,000 images and were scaled proportionally to each dataset size.

**LR decay steps at 70% and 90% of training:** A variant of the standard step decay schedule used in Detectron2's 3x config, with decay applied later to allow longer training at the full learning rate before reduction (appropriate for smaller datasets).

**Batch size 2:** A hardware constraint. Mask R-CNN with ResNet-50-FPN at 512×512 fits batch size 2 comfortably on a single 24GB GPU. Detectron2's documentation explicitly covers single-GPU training at this batch size as a standard configuration.

**Image resolution 512×512:** Consistent with Tremblay et al. (2018), who used 512×512 in their synthetic data experiments, and sufficient to resolve the geometric detail needed for 7-class brick classification.

---

## Dataset

The synthetic datasets generated by this pipeline are publicly available on Hugging Face. All four training versions are published separately to document the full iterative development process.

| Version | Number of Images | Number of Classes | Description |
|:---------:|:--------:|:---------:|-------------|
| v1 | 1,000 | 1 | Baseline: no distractor objects, 29 domain randomization axes |
| v2 | 1,000 | 1 | Distractor objects across 3 domain randomization axes added (bringing total axes from 29 to 32), 13 additional textures |
| v3 | 3,000 | 7 | 7-class scale-up |
| v4 | 7,000 | 7 | Full-scale primary dataset |

Each version includes:
- 512×512 PNG renders
- COCO-format annotations JSON (bounding boxes + instance segmentation masks)

**Dataset:** [nathankimnguyen412/houdini-lego-sdg](https://huggingface.co/datasets/nathankimnguyen412/houdini-lego-sdg)

**License:** The synthetic dataset is released under [CC BY 4.0](LICENSE). The generation pipeline is proprietary.

**Citation:** See [CITATION.cff](CITATION.cff)

---

## Training Code

The Detectron2 training and evaluation scripts are in [`/training`](training/).

---

## References

- Tremblay et al. (2018). "Training Deep Networks with Synthetic Data: Bridging the Reality Gap by Domain Randomization." CVPR Workshop. [arXiv:1804.06516](https://arxiv.org/abs/1804.06516)
- Prakash et al. (2019). "Structured Domain Randomization: Bridging the Reality Gap by Context-Aware Synthetic Data." ICRA. [arXiv:1810.10093](https://arxiv.org/abs/1810.10093)
- Goyal et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." [arXiv:1706.02677](https://arxiv.org/abs/1706.02677)
- He et al. (2019). "Rethinking ImageNet Pre-Training." ICCV. [arXiv:1811.08883](https://arxiv.org/abs/1811.08883)
- Wu et al. (2019). "Detectron2." Facebook AI Research. [GitHub](https://github.com/facebookresearch/detectron2)
- LDraw.org. "LDraw™ is an open standard for LEGO CAD programs" [ldraw.org](https://www.ldraw.org)

---

## About

Built by [Nathan Nguyen](https://github.com/nathankimnguyen412) — a pipeline engineer with nearly a decade of experience in content generation pipelines, procedural generation, and Houdini — transitioning into synthetic data generation and robotics simulation.

I am actively seeking pipeline engineering roles focused on synthetic data, embodied AI, and simulation. I am also open to research engineering collaborations, particularly in areas where procedural generation, domain randomization, and multi-modal sensor data intersect.

[LinkedIn](https://www.linkedin.com/in/nathanknguyen) | [GitHub](https://github.com/nathankimnguyen412) | [Dataset on Hugging Face](https://huggingface.co/datasets/nathankimnguyen412/houdini-lego-sdg)