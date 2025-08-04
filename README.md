## Uncertainty-Aware 3DGS-Based Pose Estimation

Author

Yu Zhou (Ph.D. Student, University of Notre Dame)

This project builds upon the variational 3DGS model developed by Ruiqi Li.

### 📌 Overview

This repository contains the implementation for our paper:

“Uncertainty-Aware Active Perception with 3D Gaussian Splatting for Robot Pose Estimation”

We propose a novel active perception pipeline that quantifies both epistemic and aleatoric uncertainty in 3D Gaussian Splatting (3DGS)-based pose estimation and leverages them for uncertainty-aware path planning in robot navigation. Unlike prior 3DGS methods that focus on rendering or localization alone, our approach enables robots to identify informative viewpoints for re-observation—without requiring real-time localization.

### 🔧 Key Features
	•	3DGS-based Pose Estimation using image comparison and optimization.
	•	Aleatoric & Epistemic Uncertainty Estimation from predicted images and scene perturbations.
	•	Information-Gain Driven Path Planning for active perception in unknown environments.
	•	Validated in both simulation and underwater real-world experiments.

### 📁 Modules
	•	variational-3dgs: Implementation of our uncertainty-aware Gaussian Splatting variant.
	•	Pose_wise_UQ: Code for estimating per-pose uncertainty using image sampling and PnP.
	•	3DGS_Dataset_Creation: Scripts to prepare datasets from real-world or simulated robot runs.

### 🎯 Applications
	•	Uncertainty-aware visual pose estimation
	•	Active visual exploration under noisy or partial observations
	•	Underwater robotic navigation with limited localization



<img width="712" height="506" alt="pose_est" src="https://github.com/user-attachments/assets/75ac0b48-7da2-45e9-9395-65571a6c6f98" />

