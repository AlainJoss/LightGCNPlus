import torch

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_u, N_v = (10000, 1000)

TRAIN_PATH = "../data/raw_data/train.csv"
SUBMISSION_PATH = "../data/submission_data/submission_users_items.pkl"

VAL_SIZE = 0.01019582787  # 12000 / 1176952