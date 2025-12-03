"""
Final evaluation on server-side test split.
Computes MAE, RMSE, MAPE.
"""

import torch
from src.fl.utils.metrics import mae, rmse, mape
from src.fl.data.loader import load_test_loader_for_server


def evaluate_final_model(model, dataset):
    """Compute metrics on test set."""
    loader = load_test_loader_for_server(dataset)

    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for X, y in loader:
            out = model(X)
            preds.append(out)
            trues.append(y)

    preds = torch.cat(preds)
    trues = torch.cat(trues)

    return {
        "MAE": mae(preds, trues),
        "RMSE": rmse(preds, trues),
        "MAPE": mape(preds, trues),
    }
