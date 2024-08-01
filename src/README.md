# Source Code

## Overview
This folder contains the essential components required to preprocess data, define models, train them, and perform post-processing for our experiments. Each file serves a specific purpose in the workflow. Below is a brief description of each file and its main functionalities.

## Files and Purposes

- **`__init__.py`**
  - Initializes the `src` module.
  - Ensures that the directory is treated as a Python package.

- **`config.py`**
  - Contains configuration settings and constants.
  - Defines parameters such as `DEVICE`, `N_u`, `N_v`, `VAL_SIZE`, and other essential constants used throughout the project.

- **`load.py`**
  - Functions to load and preprocess the data.
  - Handles data extraction, transformation, and loading into the appropriate formats for training and evaluation.

- **`models.py`**
  - Defines the architecture of the models used for matrix completion.
  - Includes the implementation of the LightGCNPlus model and functions for saving and loading models.

- **`postprocess.py`**
  - Functions for post-processing the model outputs.
  - Handles tasks such as result analysis, evaluation metrics computation, and visualization.

- **`preprocess.py`**
  - Preprocessing steps required before training the models.
  - Includes data normalization, creation of adjacency matrices, and splitting datasets into training and validation sets.

- **`train.py`**
  - Handles the training loop for the models.
  - Includes functions for training the models, computing loss, and updating model parameters.

## Usage
All functions across these files are utilized to perform experiments, train models, and fine-tune them in the `experiments` folder. Each script plays a vital role in ensuring the smooth execution of the workflow, from data preprocessing to model training and evaluation.
