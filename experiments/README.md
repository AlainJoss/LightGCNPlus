# Experiments

## Overview
This folder contains all scripts used for experimenting with different models and configurations. Each notebook serves a specific purpose in the workflow, from exploratory data analysis to training and evaluating various models. Below is a brief description of each notebook and its main functionalities.

## Files and Purposes

- **`EDA Notebook`**
  - Performs an exploratory data analysis on the provided dataset for the competition.
  - Helps understand the data distribution and check for any patterns or anomalies.

- **`Baselines Notebook`**
  - Trains baseline models (ALS and NCF) and evaluates their performance.
  - Provides a benchmark for comparison with our models.

- **`LightGCNPlus Notebook`**
  - Introduces the LightGCNPlus model variations.
  - Trains models on different configurations (variations of architectures and hyperparameters).
  - Retrains models on the best configuration based on validation scores.
  - Post-processes predictions and tests them against the Kaggle leaderboard.

- **`General Experiments Notebook`**
  - Performs general experiments with different models and configurations.

- **`Hyperparameters Notebook`**
  - Displays the results of training runs varying one hyperparameter at a time.
  - Shows the effect of each hyperparameter on model performance.

- **`Ensemble Notebook`**
  - Finds best combination of models to create an ensemble.
  - Aims to reduce the variance of predictions and improve overall performance on the public leaderboard.