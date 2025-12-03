"""
Final evaluation of the global model on held-out test data.
Computes MAE, RMSE, and MAPE for the chosen dataset.
"""

import torch

from ..models.gru_model import GRUModel
from ..utils.metrics import mae, rmse, mape
from ..data.loader import load_test_loader_for_server
from .utils_server import get_batch_size


def evaluate_final_model(state_dict, dataset):
    """Evaluate a global model state_dict on the test split."""
    batch_size = get_batch_size()
    test_loader = load_test_loader_for_server(dataset, batch_size=batch_size)

    # Infer architecture from the state_dict
    num_nodes = state_dict["fc.weight"].shape[0]
    hidden_size = state_dict["fc.weight"].shape[1]

    model = GRUModel(num_nodes=num_nodes, hidden_size=hidden_size)
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
