# DoCIntroToMl

Decision Tree Learning for WiFi Room Classification

Report link: https://www.overleaf.com/project/68f0e711f733c09457ad6042

## Project Overview

This project implements a **Decision Tree Classifier** to predict indoor room locations using WiFi signal strength measurements.
The goal is to explore how well decision tree learning can separate spatial classes from continuous signal data under **clean** and **noisy** conditions.


## Methodology

### **1. Model**

* **Algorithm:** Decision Tree Classifier implemented using **Information Gain** (Entropy Reduction) as the splitting criterion.
* **Implementation**:

  * `DecisionTree` class handles training (`decision_tree_learning`), prediction (`predict`), pruning (`prune`), and visualisation (`draw_tree`).
  * Uses a lightweight recursive `Node` dataclass to represent tree structure (`attribute`, `value`, `left`, `right`, `terminal`).
* **Stopping Conditions:**

  * Minimum information gain threshold to prevent overfitting (currently set to 0).
  * Early stopping when all samples in a node belong to the same class.
* **Pruning:** Reduced-error prunnig with a validation set to reduce overfitting.

### **2. Data**

* **Source:** WiFi signal strength datasets (`clean_dataset.txt`, `noisy_dataset.txt`).
* **Target:** Room label corresponding to each observation.

### **3. Evaluation**

* **Cross-Validation:**

  * Implemented in the `KFoldValidator` class, which manages dataset loading, shuffling, and **k-fold cross-validation**.
  * Each fold trains a new `DecisionTree`, evaluates it on the test-fold, and computes the performance metrics.

* **Pruning:**

  * Applied **post-pruning** techniques to reduce overfitting, especially on noisy data.
  * When pruning, an **internal k-fold validation** is performed per training fold.
  * **Reduced-error** pruning is applied based on validation accuracy to balance model complexity and generalization.

* **Classification Metrics**
  * Evaluation uses confusion matrices to compute:
    * Overall **accuracy**.
    * **Precision**, **recall**, and **F1-score** per class.
  * Results across folds are averaged.
  * The average model depth can be computed to assess model complexity.


## Project Structure

```
.
├── README.md
├── requirements.txt            # Python dependencies
├── main.py                     # Main entry point with evaluation and training logic
├── decision_tree.py            # Decision Tree implementation
├── validation.py               # Validation logic for model evaluation
├── report_plots.ipynb          # Jupyter notebook for generating tree and confusion matrix plots
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

### Running the Script

```bash
   python3 main.py
```

## What the Script Does

When you run `main.py`, it will:
  * Run 10-fold cross-validatio. For each outer fold, one split is used as the test set (~10%) and the remaining data (~90%) is used for training.
  * From the train split, multiple pruned `DecisionTree` models are created via internal cross-validation: each internal split trains a tree, prunes it using its validation split, and then evaluates that pruned tree on the outer test fold.
  * After all 10 folds, the script reports the averaged confusion matrices, accuracy, and per calss precision, recall, and F1-score.

## Jupyter notebook

Our plots for the report were generated using the code in `report_plots.ipynb`, additionally we have also added plots to demonstrate the effect pruning has on the size and accuracy of the decision trees.

## Dependencies

- **numpy** (2.1.1)
- **matplotlib** (3.9.2)