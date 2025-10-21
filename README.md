# DoCIntroToMl

Decision Tree Learning for WiFi Room Classification

Report link: https://www.overleaf.com/project/68f0e711f733c09457ad6042

## Project Overview

This project implements a decision tree classifier to predict room locations based on WiFi signal strength measurements. The implementation includes:

- Decision tree learning algorithm with information gain splitting
- Training and evaluation on clean and noisy datasets
- Tree visualization using matplotlib
- Performance metrics and accuracy reporting

## Project Structure

```
.
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── Jeremy/
│   ├── main.py                # Main entry point with training/evaluation
│   └── treeNode.py            # TreeNode class definition
└── wifi_db/
    ├── clean_dataset.txt      # Clean WiFi signal dataset
    └── noisy_dataset.txt      # Noisy WiFi signal dataset
```

## Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd DoCIntroToMl
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Code

### Basic Usage

To run the main decision tree training and evaluation:

```bash
cd Jeremy
python main.py
```

### What the Script Does

When you run `main.py`, it will:

1. **Load the clean dataset** from `wifi_db/clean_dataset.txt`
2. **Split data** into training (75%) and test (25%) sets
3. **Train a decision tree** using information gain
4. **Save the tree visualization** to `tree.txt` in the project root
5. **Evaluate on clean data** and print accuracy
6. **Load the noisy dataset** from `wifi_db/noisy_dataset.txt`
7. **Train and evaluate** on the noisy data
8. **Print noisy dataset accuracy**

## Development

To modify the decision tree implementation:
- Edit `Jeremy/main.py` for the main algorithm and utilities
- Edit `Jeremy/treeNode.py` for the tree node structure

To add new features:
- Implement new functions in `main.py`
- Call them from the `_demo()` function or create your own entry point

## Dependencies

- **numpy** (2.1.1)
- **matplotlib** (3.9.2)
- **scipy** (1.14.1)
