# Optimizing Neural Networks for Structured Data: A Case Study

![Python](https://img.shields.io/badge/Python-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-Learn](https://img.shields.io/badge/Sklearn-1.0-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Project Overview
This project explores the application of **Deep Neural Networks (MLP)** on structured tabular data. Using the famous **UCI Forest CoverType dataset**, the goal was to predict the forest cover type (7 classes) based on cartographic variables like elevation, slope, and soil type. 

The project focuses on **engineering a robust architecture** using Batch Normalization, Dropout, and advanced Optimization strategies, and strictly compares the results against a **Random Forest** baseline.

## Key Results

| Model Architecture | Accuracy | F1-Score (Weighted) |
| :--- | :--- | :--- |
| **Deep MLP (Final)** | **91.60%** | **0.92** |
| Random Forest (Ensemble) | 95.33% | 0.95 |
| Baseline MLP | ~86.00% | 0.85 |

> **Insight:** While the Neural Network achieved a strong 91.6% accuracy, the Random Forest proved superior for this specific tabular dataset, efficiently capturing the rule-based nature of cartographic features.

## Technical Implementation

### 1. Data Pipeline
* **Ingestion:** Robust manual fetching via Pandas (bypassing `sklearn` 403 errors)
* **Preprocessing:** Stratified Train/Test split (80/20) and `StandardScaler` for feature normalization.

### 2. Deep Learning Architecture
The final model utilizes a **Funnel Architecture** with the following engineering choices:
* **Layers:** 256 $\to$ 128 $\to$ 64 Neurons.
* **Regularization:** L2 Regularization + Dropout (0.3, 0.2, 0.1) to prevent overfitting.
* **Stabilization:** **Batch Normalization** applied after each dense layer.
* **Optimization:** `Adam` optimizer with `ReduceLROnPlateau` and `EarlyStopping`.

### 3. Comparison
* Implemented a **Random Forest Classifier** (`n_estimators=100`) on unscaled data to demonstrate the "Simpler is sometimes Better" principle in Data Science.