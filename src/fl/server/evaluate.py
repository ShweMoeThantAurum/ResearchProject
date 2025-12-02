"""
Final evaluation of the global model on held-out test data.
Computes MAE, RMSE, and MAPE for the chosen dataset.
"""

import torch

from ..models.gru_model import GRUModel
from ..utils.metrics import mae, rmse, mape
from ..data.loader import load_test_loader_for_server
from .utils_server import get_batch_size


def evaluate_global_model(state_dict, dataset):
    """Run evaluation on the roadside test partition."""
    batch_size = get_batch_size()
    test_loader, num_nodes = load_test_loader_for_server(dataset, batch_size)

    model = GRUModel(num_nodes=num_nodes, hidden_size=state_dict["decoder.weight"].shape[1])
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
    """
    Backwards-compatible wrapper.
    Internally calls evaluate_global_model().
    """
    return evaluate_global_model(state_dict, dataset)
