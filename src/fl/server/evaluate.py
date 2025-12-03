"""
Final evaluation of the global model on held-out test data.
Computes MAE, RMSE, and MAPE for the chosen dataset.
"""

import torch

from ..models.gru_model import GRUModel
from ..utils.metrics import mae, rmse, mape
from ..data.loader import load_test_loader_for_server
from ..config.settings import settings


def evaluate_global_model(state_dict, dataset):
    """Evaluate global model on the test split for a dataset."""
    batch_size = settings.batch_size
    test_loader = load_test_loader_for_server(dataset, batch_size=batch_size)

    model = GRUModel(hidden_size=settings.hidden_size)
    model.load_state_dict(state_dict)
    model.eval()

    preds = []
    trues = []

    with torch.no_grad():
        for X, Y in test_loader:
            out = model(X)
            preds.append(out)
            trues.append(Y)

    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)

    results = {
        "MAE": mae(preds, trues),
        "RMSE": rmse(preds, trues),
        "MAPE": mape(preds, trues),
    }

    return results


def evaluate_final_model(state_dict, dataset):
    """Wrapper kept for clarity with server_main usage."""
    return evaluate_global_model(state_dict, dataset)
