# Experiments

This folder contains all scripts used for experimenting with different models and configurations.

- In the EDA notebook we perform an exploratory data analysis on the provided dataset for the competition.

- In the GCMC and LightGCNPlus notebooks we introduce the corresponding models, we train them on different configurations (variations of architectures and hyperparameters), and re-train them on the configuration that achieves the best validation score. We then post-process the predictions and test them against the Kaggle leaderboard.

- In the general_experiments notebook we test different hyperparameters in isolation on both the GCMC and LightGCNPlus models, to understand their impact on performance, to reduce the search space for the hyperparameter optimization.

- In the ensamble notebook we train different models on different configurations and combine them, with the aim of reducing the variance of the predictions, and thus improving the overall performance.