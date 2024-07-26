import torch

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_u, N_v = (10000, 1000)

TRAIN_PATH = "../data/raw_data/train.csv"
SUBMISSION_PATH = "../data/submission_data/submission_users_items.pkl"
BEST_VAL_MODEL_PATH = "../data/models/best_val_model.pth"

VAL_SIZE = 0.01019582787  # 12000 / 1176952