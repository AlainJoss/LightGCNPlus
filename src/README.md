# Source Code

This folder stores all functionality used in our experiments (training and comparison of different models with different hyperparameters in the experiments folder),
which we distribute across the following modules:
- init: needed for packaging the modules in the src folder, making them accessible from the experiments folder.
- config: saves constants used throughout the program.
- load: enables to load the training data and the submission indices.
- models: defines a general model class and different subclasses.
- postprocess: enables to get the predictions from the model corresponding to the best validation, finalize them for submission, and create the submission file.
- preprocess: enables to convert the raw data to a format ingestible by the models.
- train: enables to train a model and to report results during and after training.
