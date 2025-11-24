# A Nonlinear Multi-Objective Prediction Strategy for Small-Sample Datasets in Homogeneous Catalysis

This repository provides the **MATLAB implementation** of **PSO_CRP**, a novel, nonlinear, and multi-objective prediction-capable machine learning (ML) framework introduced in the paper: 
**"A Nonlinear Multi-Objective Prediction Strategy for Small-Sample Datasets in Homogeneous Catalysis."**

## Why PSO_CRP?

Homogeneous catalyst development is often bottlenecked by expensive experiments and computationally intensive simulations—especially when only **small, sparse datasets** are available. While most ML approaches require large data volumes, **PSO_CRP is purpose-built for small-sample regimes**, leveraging simple RDKit-derived molecular descriptors (no DFT needed) to model **complex, nonlinear relationships** across **multiple reaction objectives**.

Inspired by Sigman’s linear modeling paradigm, PSO_CRP extends this vision into a **nonlinear, multi-objective, and interpretable** workflow that delivers high predictive accuracy with small-sample dataset.

## Overview

The code is organized into five main components:

1. **Data** – Preprocessed small-sample catalytic reaction datasets.
2. **Categorical Prediction (PSK-CAC)** – A nonlinear categorical prediction framework designed to predict reaction categories (e.g., conversion or yield categories).
3. **Regression Prediction (PSG-MRC)** – A hybrid regressor was designed to assess enantioselectivity or site selectivity. PSO tunes the Gaussian Process Regression (GPR) kernel, and a subsequent Multiple Linear Regression (MLR) step corrects the residuals from GPR, with the final prediction obtained as the sum of the two components.
4. **Baseline Models** – Traditional or simplified models used for performance comparison.
5. **PFI_PDP** – Implementation of **Permutation Feature Importance (PFI)** and **Partial Dependence Plots (PDP)** to enhance model explainability.

All scripts are implemented in **MATLAB** and designed to work with limited data scenarios commonly encountered in homogeneous catalysis research.

> This framework enables robust, interpretable, and forward-looking predictions even when training data is extremely scarce—a common reality in homogeneous catalysis research.
