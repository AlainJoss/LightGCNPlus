# CIL-Project: Movie Rating Prediction

## Project Description

This project aims to develop a robust and accurate model for predicting movie ratings based on user and movie data. It is part of the ETH Zurich Computational Intelligence Lab's collaborative filtering competition on Kaggle. The objective is to create a system that can provide personalized movie recommendations to users by analyzing patterns and preferences in historical rating data.

## Challenge Overview

A recommender system is designed to present items (e.g., books on Amazon, movies on Movielens, or music on LastFM) that are likely to interest the user. In collaborative filtering, recommendations are based on the known preferences of the user towards items as well as the preferences of other "similar" users.

For this challenge, we have acquired ratings from 10,000 users for 1,000 different items (movies). All ratings are integer values between 1 and 5 stars. The collaborative filtering algorithm is evaluated based on the prediction error, measured by the root-mean-squared error (RMSE).

## Installation

### Prerequisites

Before you begin, ensure you have met the following requirements:

* Python 3.x
* pip (Python package installer)

### Installation steps

1. Clone the repository:

    ```sh
    git clone https://github.com/AlainJoss/movie_rating_prediction.git
    cd movie_rating_prediction
    ```

2. Set up a virtual environment:

    ```sh
    python -m venv env
    source env/bin/activate
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Folder Structure

* `experiments`: Contains scripts for experimenting with different models and configurations.
* `src`: Contains source code for model training and evaluation.
* `data`: Stores data used in preprocessing, training, and post-processing phases.

## Usage

Here we provide a brief description of important source files

* load.py contains functions to gather the training and submission data according to paths laid out in config.py.
* models.py defines the architectures used in the models experimented with in this repo.
* train.py defines the entire training pipeline including training, evaluation, and reporting.
* preprocess.py contains functions to transform the rating matrix data into an adjacency matrix where the ratings are standardized as Z-scores
* postprocess.py includes functions to generate predictions and convert them to the format specified for submission to kaggle

To use the `LightGCN+` model and other experimental models in this project, refer to the Jupyter notebooks provided in the `experiments` folder. A further README can be found there, along with notebooks that include detailed examples and step-by-step instructions on data preprocessing, model training, hyperparameter tuning, and evaluation.

## Results

The performance of the models is evaluated using the root-mean-squared error (RMSE). The best model achieves an RMSE of 0.9780 on the validation set. Detailed results and visualizations are provided in the experiment notebooks.
