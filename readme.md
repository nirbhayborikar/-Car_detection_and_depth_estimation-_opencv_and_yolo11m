# Car Detection & Depth Estimation using YOLOv11m and KITTI

This project presents a camera-based system for **car detection** and **monocular depth estimation** using **YOLOv11m** and a curated subset of the **KITTI Vision Benchmark Suite**. The pipeline detects cars, evaluates detection accuracy, and estimates distances using a calibrated pinhole-camera model.

🔗 **Project Page:** https://nirbhayborikar.github.io/projects/03-computervision-car-detection-yolo  
🔗 **Repository:** https://github.com/nirbhayborikar/-Car_detection_and_depth_estimation-_opencv_and_yolo11m

---

## Overview

This project investigates the capabilities and limitations of monocular vision for autonomous driving. It includes:

- Car detection with **YOLOv11m**
- Depth estimation using camera intrinsics
- Evaluation using **IoU**, **Precision**, **Recall**, **Average Precision (AP)**
- Ground-truth depth generation from KITTI geometry
- Comparison of YOLO-estimated vs. ground-truth distances

---

## KITTI Dataset

A subset of 20 stereo images from the KITTI dataset was used. Each sample includes:

- Left RGB image  
- Bounding box labels  
- Camera calibration matrix  
- Derived per-object ground-truth depth  

Since KITTI does not directly provide depth per object, distances were computed using:

- Four bounding-box corners  
- Four midpoint locations  
- Ground truth = **minimum** of the 8 geometric distances  

This provides consistent reference distances for evaluation.

---

## Methodology

### 1. Car Detection (YOLOv11m)
- Pretrained YOLOv11m model used for “car” class detection.
- Bounding boxes and confidence scores extracted.
- IoU used to filter detections and match ground truth.

### 2. Object Detection Evaluation
- IoU threshold: **λ = 0.75**
- IoU ≥ λ → True Positive  
- IoU < λ → False Positive  
- Detections sorted by confidence to compute Precision–Recall curves and AP.

### 3. Depth Estimation
Depth is computed from camera geometry:
Z= (h⋅fy​)​ / ∣ybase​−cy​∣
Where:
 h — camera height
 fy- vertical focal length
 ybase — car base pixel position
 cy — principal point (image center)


Depth is accurate up to ~40 m, with declining precision beyond 40–70 m.

---

## Results

- YOLOv11m provides consistent detection across all KITTI frames.
- Depth estimation closely matches ground truth for near and mid-range objects.
- Performance decreases at long distances due to monocular limitations.
- Repository includes:
  - Precision–Recall plots  
  - IoU-filtering visualizations  
  - Depth correlation plots  



---

## Future Work

- Fine-tune YOLO on KITTI for improved small-object detection  
- Integrate stereo depth as comparison baseline  
- Add multi-object tracking  
- Deploy pipeline for real-time inference  

---

## Author

**Nirbhay Borikar**  
Portfolio: https://nirbhayborikar.github.io/

#Note: refer to data.md to know which folder includes what
