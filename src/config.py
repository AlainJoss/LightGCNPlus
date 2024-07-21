import torch

DEVICE = torch.device("mps") if torch.device("mps") else torch.device("cpu")
N_u, N_v = (10000, 1000)

TRAIN_PATH = "data/raw_data/train.csv"
SUBMISSION_PATH = "data/raw_data/sample_submission.csv"

