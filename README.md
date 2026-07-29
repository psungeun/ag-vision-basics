# 🌱 Plant Phenotyping & Computer Vision Basics

A foundational computer vision practice repository for agricultural data analysis and plant phenotyping research.

<div align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=OpenCV&logoColor=white"/>
  <img src="https://img.shields.io/badge/Plant_Phenotyping-4CAF50?style=flat-square&logo=leaf&logoColor=white"/>
  <img src="https://img.shields.io/badge/Precision_Agriculture-00599C?style=flat-square&logo=agri&logoColor=white"/>
</div>

<br>

## 📌 Overview
This repository records and tests image processing techniques applicable to real-world agricultural environments, such as smart farms and autonomous agricultural machinery. It is built upon the core algorithms covered in the undergraduate course "Plant Phenotyping and Imaging."

## 🛠 Contents (To-Do)
- [x] **Tractor Object Matching:** Object detection testing for autonomous agricultural machinery (e.g., John Deere) using SIFT & RANSAC algorithms.
- [ ] **Leaf Area Segmentation:** Crop leaf segmentation and area extraction in complex backgrounds using the GrabCut algorithm.
- [ ] **Image Preprocessing:** Image preprocessing for irregular lighting correction in greenhouse and open-field environments (Gaussian Blur, Gamma Correction).

---

## 📝 Project 01: Robust Tractor Recognition using SIFT & RANSAC

In agricultural environments, recognizing specific machinery or plant structures is highly challenging due to dynamic background noise, varying scales, and different viewpoints. 

This project demonstrates how to reliably detect and match a 3D object (a John Deere 8R 410 tractor) across different field images by extracting localized features and mathematically filtering out false matches.

### 🔍 Methodology

1. **ROI Extraction (Cropping)**
   - **Why?** Real-world agricultural images contain immense background noise (skies, fields, structures). By cropping the model image to a specific Region of Interest (ROI) containing the tractor, we prevent the algorithm from extracting features from irrelevant backgrounds, drastically increasing matching accuracy.
   
2. **Keypoint Detection (SIFT)**
   - Utilized the **Scale-Invariant Feature Transform (SIFT)** algorithm to extract keypoints and 128-dimensional descriptors. SIFT ensures that the tractor's features (such as the grill pattern and logos) are recognized regardless of image scale or rotation differences.

3. **Feature Matching (FLANN & Ratio Test)**
   - Employed a **FLANN-based KD-Tree matcher** for fast approximate nearest neighbor searches.
   - Applied **Lowe's Ratio Test** (Threshold = 0.7) to eliminate ambiguous or poorly matched putative correspondences.

4. **Geometric Verification (RANSAC)**
   - Since initial matches often contain false positives (outliers), the **RANSAC (Random Sample Consensus)** algorithm was applied to estimate a spatial transformation model (Homography). RANSAC robustly discards geometrically incorrect matches, leaving only highly reliable inliers.

### 🚀 Results
* **Model Keypoints:** Successfully extracted from the cropped ROI.
* **Scene Keypoints:** Extracted from the full, complex field environment.
* **RANSAC Inliers:** The final output demonstrates highly accurate mapping between the cropped model features and the target scene, completely ignoring background artifacts like clouds or soil.

<br>

<div align="center">
  <img src="result_output.png" alt="SIFT and RANSAC Tractor Matching Result" width="90%">
</div>

---

## 📝 Project 02: Leaf Area Segmentation & Phenotypic Trait Extraction

In smart farming and precision agriculture, accurately measuring plant phenotypic traits (e.g., leaf area) is essential for monitoring crop health and growth rates. This project demonstrates how to isolate a specific target leaf from a highly cluttered greenhouse background and computationally extract its physical area.

### 🔍 Methodology

1. **Foreground Segmentation (GrabCut Algorithm)**
   * **Why?** Greenhouse images contain complex backgrounds (soil, pipes, other overlapping plants) that simple thresholding cannot handle. 
   * **Approach:** Utilized the Graph-Cut based **GrabCut** algorithm. By initializing a bounding box (ROI) around the target leaf, the algorithm models the color distribution (Gaussian Mixture Model) of the foreground and background, iteratively refining the segmentation mask to isolate only the target leaf.
   
2. **Morphological Analysis & Contour Extraction**
   * Applied binary masking to extract the segmented foreground.
   * Utilized `cv2.findContours` (External Retrieval Mode) to mathematically define the boundary of the isolated leaf.

3. **Phenotypic Trait Calculation (Leaf Area)**
   * Computed the total pixel area of the leaf using image moments (`cv2.contourArea`). 
   * This non-destructive measurement technique is a fundamental step toward automated plant growth monitoring systems.

### 🚀 Results
* **Target Isolation:** Successfully removed overlapping leaves, flowers, and greenhouse structures, leaving a clean mask of the target strawberry leaf.
* **Trait Extraction:** The algorithm automatically drew a bounding contour around the target leaf and accurately calculated its spatial area in pixels, providing quantifiable phenotypic data.
