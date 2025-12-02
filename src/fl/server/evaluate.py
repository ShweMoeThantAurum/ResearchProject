"""
Final evaluation of the global model.
"""

import numpy as np
import torch

from src.fl.utils.metrics import mae, rmse, mape
from src.fl.data.loader import load_full_eval_dataset
from src.fl.models import SimpleGRU
from src.fl.config import settings


def evaluate_final_model(state):
    """
    Evaluate final global model on full evaluation set.
    """
    x, y = load_full_eval_dataset()

    model = SimpleGRU(
        input_dim=1,
        hidden_size=settings.get_hidden_size(),
        num_layers=1
    )

    model.load_state_dict(state)
    model.eval()

    x_t = torch.tensor(x).float()
    preds = model(x_t).detach().cpu().numpy().flatten()

    return {
        "MAE": mae(y, preds),
        "RMSE": rmse(y, preds),
        "MAPE": mape(y, preds)
    }
