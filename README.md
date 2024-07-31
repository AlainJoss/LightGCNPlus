# CIL-Project: Movie Rating Prediction

## Project Description
This project aims to develop a robust and accurate model for predicting movie ratings based on user and movie data. It is part of the ETH Zurich Computational Intelligence Lab's collaborative filtering competition on Kaggle. The objective is to create a system that can provide personalized movie recommendations to users by analyzing patterns and preferences in historical rating data.

## Challenge Overview
A recommender system is designed to present items (e.g., books on Amazon, movies on Movielens, or music on LastFM) that are likely to interest the user. In collaborative filtering, recommendations are based on the known preferences of the user towards other items, and also take into account the preferences of other users.

For this challenge, we have acquired ratings from 10,000 users for 1,000 different items (movies). All ratings are integer values between 1 and 5 stars. The collaborative filtering algorithm is evaluated based on the prediction error, measured by the root-mean-squared error (RMSE).

## Installation 
### Prerequisites
Before you begin, ensure you have met the following requirements:
* Python 3.x
* pip (Python package installer)

### Installation steps
1.  Clone the repository:
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
- `experiments`: Contains scripts for experimenting with different models and configurations.
- `src`: Contains source code for model training and evaluation.
- `data`: Stores data used in preprocessing, training, and post-processing phases.

## Usage
To use the `LightGCN+` model and other experimental models in this project, refer to the Jupyter notebooks provided in the `experiments` folder. These notebooks include detailed examples and step-by-step instructions on data preprocessing, model training, hyperparameter tuning, and evaluation.

## Results
The performance of the models is evaluated using the root-mean-squared error (RMSE). The best model achieves an RMSE of X.XXX on the validation set. Detailed results and visualizations are provided in the experiment notebooks.