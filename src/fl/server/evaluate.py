"""
Final evaluation of global model after training.
"""

import numpy as np
import torch

from src.fl.models import SimpleGRU
from src.fl.utils.metrics import mae, rmse, mape


def evaluate_final_model(global_state, proc_dir, num_nodes, hidden):
    """
    Load test dataset, run model, compute metrics.
    """
    X_test = np.load(proc_dir + "/X_test.npy")
    y_test = np.load(proc_dir + "/y_test.npy")

    device = torch.device("cpu")

    model = SimpleGRU(num_nodes=num_nodes, hidden_size=hidden).to(device)
    model.load_state_dict(global_state)
    model.eval()

    X_t = torch.from_numpy(X_test).float().to(device)
    with torch.no_grad():
        preds = model(X_t).cpu().numpy()

    y_true = y_test

    return {
        "MAE": mae(preds, y_true),
        "RMSE": rmse(preds, y_true),
        "MAPE": mape(preds, y_true),
    }
