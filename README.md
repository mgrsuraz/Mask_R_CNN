# Instance Segmentation with Mask R-CNN

This repository implements **instance segmentation** using **Mask R-CNN** to detect and segment everyday objects.  
The model is **fine-tuned on a custom dataset** and deployed using a **Gradio web interface** for interactive inference.

---

## Overview

**Instance segmentation** is a computer vision task that:
- Detects objects
- Classifies each object
- Generates a **pixel-level mask** for every object instance

Unlike object detection or semantic segmentation, instance segmentation distinguishes **individual objects of the same class** with precise boundaries.

---

## Target Classes
The model segments the following objects:
- **Pen** — small, thin object
- **Plug** — medium-sized object with irregular shape
- **Shoe** — large object with high visual variation

These classes were chosen to evaluate performance across **different object scales and shapes**.

---

## Dataset
- **Type:** Custom, self-collected dataset
- **Images:** 200+ real-world images captured using a smartphone
- **Annotations:** Polygon-based instance masks (JSON)
- **Classes:** Pen, Plug, Shoe (+ background)

### Dataset Split
| Split | Percentage |
|------|------------|
| Train | 80% |
| Val  | 10% |
| Test | 10% |

---
Link to dataset: https://drive.google.com/drive/folders/1rgXpOqdZQEemoX1MMP9XY8V30HNnJDkH?usp=drive_link

## Annotation Format
Each image has a corresponding JSON annotation file.

```json
{
  "key": "00167.jpg",
  "width": 4284,
  "height": 5712,
  "boxes": [
    {
      "type": "polygon",
      "label": "plug",
      "points": [[x1,y1], [x2,y2], ...]
    }
  ]
}
````

* Each polygon defines one object instance
* Masks are converted into tensors for training Mask R-CNN

---

## Model Architecture

* **Base Model:** Mask R-CNN
* **Backbone:** ResNet-50 + Feature Pyramid Network (FPN)
* **Key Components:**

  * Region Proposal Network (RPN)
  * RoIAlign
  * Box classification & regression head
  * Fully convolutional mask head

### Loss Function

L = L_cls + L_box + L_mask

* Classification loss
* Bounding box regression loss (Smooth L1)
* Mask loss (binary cross-entropy)

---

## Network Adaptation

The final prediction heads were modified to support custom classes.

```python
model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT")

model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features_box, num_classes=4
)

model.roi_heads.mask_predictor = MaskRCNNPredictor(
    in_features_mask, dim_reduced, num_classes=4
)
```

---

## Optimization & Hyperparameter Tuning

* **Optimizer:** AdamW
* **Batch size:** 6
* **Scheduler:** OneCycleLR

### Tuned Hyperparameters

| Parameter       | Values       |
| --------------- | ------------ |
| Learning Rate   | [1e-3, 5e-4] |
| Weight Decay    | [1e-4]       |
| Epochs (tuning) | 1            |

### Best Configuration

* **LR:** 0.0005
* **Weight Decay:** 0.0001

---

## Evaluation Metrics

* **mAP@0.50** (object detection quality)
* **Mean IoU (True Positives)** (mask accuracy)

| Model              | mAP@0.50  | Mean IoU  |
| ------------------ | --------- | --------- |
| Pre-trained (COCO) | ~0.00     | ~0.25     |
| Fine-tuned         | **~0.51** | **~0.82** |

---

## Innovation: Test-Time Augmentation (TTA)

We applied **Test-Time Augmentation**, a modern inference technique:

* Original image + horizontal flip
* Predictions merged via Non-Maximum Suppression (NMS)

**Benefits:**

* Improved robustness
* Better recall for small objects
* No retraining required

---

## Key Takeaways

* Transfer learning is essential for custom instance segmentation
* Mask R-CNN performs well even with limited data
* Hyperparameter tuning and threshold selection matter
* TTA improves robustness without retraining
* Visualization helps interpret model behavior

---

## References

* He et al., *Mask R-CNN*, ICCV 2017
* PyTorch & Torchvision Documentation
* COCO Dataset

---

## Author

**Suraj Thapa**
MS in Data Science
University of New Haven, CT, USA.
```
