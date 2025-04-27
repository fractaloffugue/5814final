# Data Sampling Method Based on Integrated Active Learning and Reinforcement Learning Agents

## Overview
This repository contains the implementation of a reinforcement learning-based framework for online active learning, inspired by the CBeAL (Contextual Bandit-Based Ensemble Active Learning) methodology. Our approach addresses the challenge of balancing exploration and exploitation during sample selection in streaming scenarios, particularly within industrial cyber-physical systems.

## Dataset
We use the UCI Spambase dataset, which includes:
- 4,601 email instances with 57 features and 1 binary target variable
- 48 word frequency attributes (continuous values)
- 6 character frequency attributes (continuous values)
- 3 capital run length attributes (continuous values)
- Binary classification problem (spam or non-spam)
- Relatively balanced class distribution (39.4% spam)

The dataset was collected in the 1990s by researchers from Hewlett-Packard Labs and is hosted by the UCI Machine Learning Repository.

## Project Structure
- `data/`: Contains the Spambase dataset
- `agents/`: Implementation of active learning agents
- `controllers/`: EXP4.P and EWMA ensemble controllers
- `models/`: MLP classifier implementation
- `utils/`: Helper functions for preprocessing and evaluation
- `experiments/`: Scripts to run and evaluate experiments

## Active Learning Agents
The framework constructs a diverse committee of active learning agents:

1. **Max-Min Distance Based Agent (ALAgent_X_MMd_B)**: Encourages uniform space exploration by selecting samples far from existing labeled data.

2. **High-Dimensional Density Agent (ALAgent_X_hD_B)**: Samples from low-density regions to uncover new cluster structures.

3. **Margin-Based Agent (ALAgent_X_M_B)**: Targets exploitation by selecting samples close to the decision boundary.

4. **Reinforced Exploitation Agent (RAL_B)**: Dynamically adjusts a certainty threshold using reinforcement learning principles.

5. **RAL_B_EXP4P**: Enhanced version of the Reinforced Exploitation Agent with EXP4.P weight updates.

6. **Exp4P_EN_SWAP**: Ensemble controller that combines all agent outputs with an EWMA-based flipping mechanism.

## Ensemble Controller
The EXP4.P-EWMA ensemble controller integrates the agents via:
- Soft voting system that adapts based on reward feedback
- EWMA-based flipping mechanism to prevent agent dominance
- Dynamic adaptability to ensure efficient sample selection

## Reward Function
The reward signal is:
- Positive (+2) when a selected sample would have been misclassified
- Negative (-1) when a selected sample provides no new information
- This design encourages informative sample acquisition

## Experimental Setup
- **Preprocessing**: Standardization, stratified train-test split (80-20), class balance
- **Model**: Two-layer MLP with ReLU activation and sigmoid output
- **Training**: Adam optimizer with early stopping
- **Hyperparameter Tuning**: Stratified k-fold cross-validation (k=3) with F1 score objective

## Key Results
| Method | # Samples | Accuracy | F1 Score |
|--------|-----------|----------|----------|
| Agent2 (al1 + al2) | 1324 | 0.9349 | 0.9174 |
| Agent4 (al1 + al2 + al3 + rl1) | 1939 | 0.9240 | 0.9059 |
| Agent6 (all agents) | 2103 | 0.9316 | 0.9159 |
| Baseline MLP | 2900 | 0.9251 | 0.9086 |

A compact agent ensemble (Agent2) achieves high accuracy (93.49%) and F1 score (91.74%) while labeling significantly fewer samples than full supervision.

## Dependencies
- Python 3.8+
- NumPy
- Pandas
- SciPy
- Scikit-learn
- PyTorch

## Usage
```python
# Example usage
from agents import ALAgent_X_MMd_B, ALAgent_X_hD_B
from controllers import Exp4P_EN_SWAP
from models import MLPClassifier

# Initialize agents
al1 = ALAgent_X_MMd_B(input_dim, window_size, density_thresh)
al2 = ALAgent_X_hD_B(input_dim, window_size, density_thresh)

# Create ensemble
ensemble = Exp4P_EN_SWAP([al1, al2], max_T=X_train.shape[0], delta=0.1)

# Train and evaluate
results = train_and_eval(X_train, y_train, ensemble)