# yolo_v3_implementation 📦

> Rebuilding YOLOv3 from the original paper — Darknet-53, multi-scale detection, anchor decoding, NMS, and inference — implemented from scratch in Python.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Why This Exists

Most YOLO implementations hide the difficult parts behind high-level APIs:
tensor reshaping, anchor decoding, feature pyramid routing,
and post-processing logic.

This project rebuilds YOLOv3 from first principles to understand how real-time object detection systems actually work internally.

The goal is not production deployment — it is architectural understanding.

---

## What’s Implemented

* Darknet-53 backbone network
* Residual convolutional blocks
* Multi-scale detection heads (13×13, 26×26, 52×52)
* Anchor box prediction and decoding
* Bounding box regression
* Objectness confidence prediction
* Class probability prediction
* IoU computation
* Non-Maximum Suppression (NMS)
* End-to-end inference pipeline

---

## Architecture Breakdown

| Component             | Purpose                              |
| --------------------- | ------------------------------------ |
| Darknet-53            | Feature extraction backbone          |
| Residual blocks       | Stable deep feature learning         |
| Detection heads       | Multi-scale object prediction        |
| Anchor boxes          | Bounding box priors                  |
| IoU + NMS             | Remove overlapping detections        |
| Bounding box decoding | Convert raw outputs into coordinates |

---

## Quick Start

```bash
git clone https://github.com/ashutoshsharmadev/yolo_v3_implementation.git
cd yolo_v3_implementation

pip install torch torchvision numpy matplotlib jupyter

jupyter notebook
```

Open the notebook and run cells sequentially.

---

## Implementation Details

This project manually reconstructs key detection mechanics normally abstracted away by frameworks:

* Tensor reshaping from convolution output → detection format
* Bounding box decoding relative to anchor priors
* Feature pyramid routing across 3 prediction scales
* Confidence thresholding and NMS filtering
* Raw tensor → final bounding box conversion pipeline

---

## What I Learned

* How anchor-based detectors predict object locations
* Why YOLO predicts at multiple spatial resolutions
* The math behind `(tx, ty, tw, th)` box encoding
* How confidence scores differ from class probabilities
* How detection tensors become final predictions

---

## Project Structure

```text
yolo_v3_implementation/
├── yolov3.ipynb         # Main implementation notebook
└── README.md
```

---

## Roadmap

* [ ] Add training loop with COCO/VOC dataset support
* [ ] Export trained weights for standalone inference
* [ ] Compare outputs against official Darknet weights
* [ ] Add visualization tools for intermediate feature maps
* [ ] ONNX export + TensorRT inference path
* [ ] Custom dataset fine-tuning support

---

## Related Projects

* [visions](https://github.com/ashutoshsharmadev/visions)
  Real-time YOLO inference pipeline using YOLOv8.

* [neural_nets_in_C](https://github.com/ashutoshsharmadev/neural_nets_in_C)
  Neural networks and backpropagation implemented from scratch in pure C.

---

## Reference

> Redmon, J., & Farhadi, A. (2018).
> *YOLOv3: An Incremental Improvement.*
> https://arxiv.org/abs/1804.02767

---

## License

MIT
