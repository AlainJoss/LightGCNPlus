import torch

DEVICE = torch.device("cpu")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps") 
elif torch.cuda.is_available():
    DEVICE = torch.device("cpu")
N_u, N_v = (10000, 1000)

TRAIN_PATH = "../data/raw_data/train.csv"
SUBMISSION_PATH = "../data/raw_data/sample_submission.csv"

# VAL_SIZE = 0.008496  # 10000 / 1176952 ~ 0.85%
VAL_SIZE = 0.05

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