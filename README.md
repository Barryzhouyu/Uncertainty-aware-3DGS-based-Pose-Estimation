## Uncertainty-Aware 3DGS-Based Pose Estimation for Underwater Robot

### Author
[Yu Zhou](https://barryzhouyu.github.io/yuzhoubarry.github.io/)

> This project builds upon the [variational 3DGS model](https://github.com/csrqli/variational-3dgs) developed by Ruiqi Li.
### 📌 Overview

This repository contains the implementation for our paper:

#### Topic: Uncertainty-aware 3D Gaussian Splatting (3DGS) based Pose Estimation for Underwater Vehicles 

We propose an uncertainty-aware 3DGS framework that estimates pose-wise uncertainty via distributions over Gaussian parameters generated from the Variantional-3dgs model, enabling the AUV to identify areas of low confidence for potential revisitation or exploration.

<div align="center">
  <img width="500" alt="3dgs_flow_chart" src="https://github.com/user-attachments/assets/54507a79-4934-48d8-a02b-e1bf842bfec9" />
</div>

### 🔧 Results
Comparison between ground truth views and rendered images from two scenes: one in a simulator and the other in an outdoor swimming pool.

<div align="center">
<img width="800" alt="rendering_result" src="https://github.com/user-attachments/assets/57b74968-47d8-4d49-89a6-e14640f8fe58" />
</div>

### 🎯 Applications
	•	Uncertainty-aware visual pose estimation
	•	Active visual exploration under noisy or partial observations
	•	Underwater robotic navigation with limited localization


