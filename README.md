# LightGCN+: A Graph Convolutional Network for Matrix Completion

## Introduction

This project introduces LightGCN+, an advanced graph convolutional encoder-decoder model for collaborative filtering in personalized recommendation systems. LightGCN+ extends the original LightGCN architecture to address the task of matrix completion, enabling the prediction of real-valued ratings.

Key features of LightGCN+:
1. Tailored for explicit rating prediction, rather than top-N recommendation (simple ranking).
2. Leverages a modified graph convolutional framework to handle real-valued rating matrices.
3. Incorporates a multi-layer perceptron (MLP) at the decoder stage.
4. Achieves robust performance comparable to state-of-the-art models.

This repository contains the implementation of LightGCN+, along with experiments conducted on the ETHZ-CIL-Collaborative-Filtering-2024 dataset. We provide tools for data preprocessing, model training, and result analysis to facilitate further research and reproducing our results.

## Repository Structure

The repository is organized into three main folders:

### Experiments

- `EDA.ipynb`: Exploratory Data Analysis - Initial data exploration and visualization.
- `baselines.ipynb`: Baseline Model Training - Training and evaluation of ALS and NCF models.
- `lightgcn_plus.ipynb`: LightGCN+ Model Training and Evaluation - Training and evaluating the LightGCN+ model.
- `general_experiments.ipynb`: General Experiments - Experiments with different configurations.
- `hyperparameter_tuning.ipynb`: Hyperparameter Tuning - Tuning hyperparameters for optimal performance.
- `ensemble.ipynb`: Ensemble Model Creation - Combining models to improve performance.

### Data

Stores all data used in the project:
- `model_state/`: Best validation states of training runs.
- `raw_data/`: Original training data.
- `submission_data/`: Submission IDs and files.

### Source Code

Contains the core implementation files:
- `config.py`: Configuration settings and constants.
- `load.py`: Data loading and preprocessing functions.
- `models.py`: Model architecture definitions, including LightGCN+.
- `postprocess.py`: Output processing and analysis functions.
- `preprocess.py`: Data preprocessing steps.
- `train.py`: Model training loop and related functions.

Each folder contains a detailed README with further information about its contents and purpose.

## Getting Started

### Prerequisites

The project requires Python 3.8 or higher. To set up the project, run the following commands:

#### macOS/Linux
```bash
git clone git@github.com:AlainJoss/LightGCNPlus.git
cd LightGCNPlus
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

#### Windows
```bash
git clone git@github.com:AlainJoss/LightGCNPlus.git
cd LightGCNPlus
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Running Experiments

To run the experiments, navigate to the `experiments/` folder and execute the desired Jupyter notebook. The notebooks contain detailed instructions and explanations for each experiment.

## Results

The LightGCN+ model demonstrates competitive performance on the ETHZ-CIL-Collaborative-Filtering-2024 dataset, achieving strong results on the Kaggle public leaderboard.

Our results, summarized in the table below, compare the performance of various configurations of LightGCN+, including the LightGCN+ ensemble, against two baseline models: Neural Collaborative Filtering (NCF) and Alternating Least Squares (ALS). We use the root mean squared error (RMSE) as evaluation metric.

### Model Performance Comparison

| **Model**        | **Configuration**          | **RMSE Score** |
|------------------|----------------------------|----------------|
| LightGCN++       | Ensemble                   | 0.9771         |
| LightGCN+        | K=28, L=4, C=(12K, 6K)     | 0.9844         |
| LightGCN+        | K=28, L=5, C=(12K, 6K)     | 0.9875         |
| NCF              |                            | 1.0940         |
| ALS              |                            | 1.4260         |

## Citations

The main inspiration for the creation of the LightGCN+ model comes from the following papers:

- [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126)
- [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)
- [Graph Convolutional Matrix Completion](https://arxiv.org/abs/1706.02263)

## Acknowledgements

This project was developed as part of the course "Computational Intelligence Lab" at ETH Zurich. We would like to thank the course instructors and teaching assistants for setting up the project.

## Authors
- Alain Joss
- Solomon Thiessen
- Antoine Suter
- Damian Cordes

Department of Computer Science, ETH Zurich, Switzerland