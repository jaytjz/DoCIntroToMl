# DoCIntroToMl

Decision Tree Learning for WiFi Room Classification

Report link: https://www.overleaf.com/project/68f0e711f733c09457ad6042

## Project Overview

This project implements a **Decision Tree Classifier** to predict indoor room locations using WiFi signal strength measurements.
The goal is to explore how well classical decision tree learning can separate spatial classes from continuous signal data under **clean** and **noisy** conditions.


## Methodology

### **1. Model**

* **Algorithm:** Decision Tree built from scratch using **Information Gain** as the splitting criterion.
* **Stopping Conditions:**

  * Minimum information gain threshold to prevent overfitting.
  * Early stopping when all samples in a node belong to the same class.
* **Tree Representation:** Implemented via a recursive `TreeNode` class, allowing easy visualization and traversal.

### **2. Data**

* **Source:** WiFi signal strength datasets (`clean_dataset.txt`, `noisy_dataset.txt`).
* **Target:** Room label corresponding to each observation.

### **3. Evaluation**

* **Cross-Validation:**

  * Implemented **10-fold cross-validation** for robust performance estimation.

* **Pruning:**

   * Applied **post-pruning** techniques to reduce overfitting, especially on noisy data.
   * Pruning decisions were based on validation set performance to balance model complexity and generalization.


## Project Structure

```
.
├── README.md
├── requirements.txt            # Python dependencies
├── main.py                     # Main entry point with evaluation and training logic
├── decision_tree.py            # Decision Tree implementation
├── validation.py               # Validation logic for model evaluation
├── report_plots.ipynb          # Jupyter notebook for generating tree and confusion matrix plots
├── evaluation.ipynb            # Jupyter notebook for evaluation experiments
└── wifi_db/
    ├── clean_dataset.txt      # Clean WiFi signal dataset
    └── noisy_dataset.txt      # Noisy WiFi signal dataset
```

## Setup

### Installation

1. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## What the Script Does

When you run `main.py`, it will:

- Run 10-fold cross-validation: each round holds out one fold as test (~10%); from the remaining 9 folds use 1 as validation (~10%) and 8 for training (~80%).
- Train a Decision Tree (Information Gain) on the training set with stopping rules; use the validation set for pruning/tuning; evaluate on the held-out test fold and record metrics.
- If skipping validation-based pruning, train on all 9 folds (~90%) and test on the held-out fold.
- Report accuracy, precision, recall, and F1-score over the 10 test folds on the clean dataset.

## Dependencies

- **numpy** (2.1.1)
- **matplotlib** (3.9.2)