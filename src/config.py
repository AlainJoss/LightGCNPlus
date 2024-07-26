import torch

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_u, N_v = (10000, 1000)

TRAIN_PATH = "../data/raw_data/train.csv"
SUBMISSION_PATH = "../data/submission_data/submission_users_items.pkl"

# VAL_SIZE = 0.008496  # 10000 / 1176952 ~ 0.85%
VAL_SIZE = 0.01

MODEL_PATH = "../data/models/best_val_model.pth"

BASE_HYPERPARAMS = {
    "L": 1,
    "K": 30,
    "INIT_EMBS_STD": 0.1,
    "LR": 0.08,
    "WEIGHT_DECAY": 1e-04,
    "EPOCHS": 500,
    "STOP_THRESHOLD": 1e-06
}

SPLITS = 2