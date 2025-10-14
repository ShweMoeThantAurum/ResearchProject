"""
Runs a simple FedAvg baseline over SZ-Taxi client splits.
"""
import os, math, time, copy, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.models.simple_gru import SimpleGRU

PROC = "data/processed/sz/prepared"
OUT = "outputs/fedavg_sz"
os.makedirs(OUT, exist_ok=True)

class ClientPaddedDataset(Dataset):
    """
    Loads a client's subset and pads to full node dimension.
    """
    def __init__(self, X_client_path, y_client_path, idxs, num_nodes):
        cX = np.load(X_client_path)  # [S, L, k]
        cY = np.load(y_client_path)  # [S, k]
        S, L, k = cX.shape
        X = np.zeros((S, L, num_nodes), dtype=np.float32)
        Y = np.zeros((S, num_nodes), dtype=np.float32)
        X[:, :, idxs] = cX
        Y[:, idxs] = cY
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i]

def get_device():
    """Selects MPS on Apple Silicon if available."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def client_indices(num_nodes, num_clients, i):
    """Returns node indices for client i (deterministic split)."""
    parts = np.array_split(np.arange(num_nodes), num_clients)
    return parts[i]

def load_global_sets():
    """Loads full train/valid/test sets for evaluation only."""
    Xtr = np.load(f"{PROC}/X_train.npy")
    ytr = np.load(f"{PROC}/y_train.npy")
    Xva = np.load(f"{PROC}/X_valid.npy")
    yva = np.load(f"{PROC}/y_valid.npy")
    Xte = np.load(f"{PROC}/X_test.npy")
    yte = np.load(f"{PROC}/y_test.npy")
    return Xtr, ytr, Xva, yva, Xte, yte

def make_eval_loader(X, y, batch):
    """Creates a DataLoader for evaluation."""
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()
    ds = torch.utils.data.TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch, shuffle=False)

def model_init(num_nodes, hidden_size=64):
    """Initializes the GRU model."""
    return SimpleGRU(num_nodes=num_nodes, hidden_size=hidden_size)

def clone_state(state):
    """Deep-copies a model state dict."""
    return {k: v.detach().clone() for k, v in state.items()}

def set_state(model, state):
    """Loads a state dict into model."""
    model.load_state_dict(state)

def average_states(states, weights):
    """Computes weighted average of state dicts."""
    avg = {}
    total = sum(weights)
    for k in states[0].keys():
        acc = None
        for s, w in zip(states, weights):
            val = s[k] * (w / total)
            acc = val if acc is None else acc + val
        avg[k] = acc
    return avg

def train_local(model, loader, device, epochs, lr):
    """Trains one client locally and returns updated state and sample count."""
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    n_samples = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            yhat = model(x)
            loss = loss_fn(yhat, y)
            loss.backward()
            opt.step()
            n_samples += x.size(0)
    return clone_state(model.state_dict()), n_samples

def evaluate(model, loader, device):
    """Evaluates MAE and RMSE over a loader."""
    model.eval()
    m_mae, m_rmse, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            m_mae += (yhat - y).abs().sum().item()
            m_rmse += ((yhat - y) ** 2).sum().item()
            n += y.numel()
    mae_val = m_mae / n
    rmse_val = math.sqrt(m_rmse / n)
    return mae_val, rmse_val

def run():
    """Runs simple FedAvg over SZ clients and logs results."""
    device = get_device()
    meta = json.load(open(f"{PROC}/meta.json"))
    num_clients = meta["num_clients"]

    # Evaluation loaders (full sets)
    Xtr, ytr, Xva, yva, Xte, yte = load_global_sets()
    eval_train = make_eval_loader(Xtr, ytr, batch=128)
    eval_valid = make_eval_loader(Xva, yva, batch=128)
    eval_test  = make_eval_loader(Xte, yte, batch=128)

    num_nodes = Xtr.shape[-1]
    model = model_init(num_nodes=num_nodes).to(device)
    global_state = clone_state(model.state_dict())

    rounds = 5
    local_epochs = 1
    batch = 64
    lr = 1e-3

    print(f"FedAvg | rounds {rounds} | clients {num_clients} | local_epochs {local_epochs}")

    for r in range(1, rounds + 1):
        t0 = time.time()
        client_states, weights = [], []

        for i in range(num_clients):
            idxs = client_indices(num_nodes, num_clients, i)
            cX = f"{PROC}/clients/client{i}_X.npy"
            cY = f"{PROC}/clients/client{i}_y.npy"
            ds = ClientPaddedDataset(cX, cY, idxs, num_nodes)
            dl = DataLoader(ds, batch_size=batch, shuffle=True)

            set_state(model, global_state)
            model.to(device)
            state_i, n_i = train_local(model, dl, device, local_epochs, lr)
            client_states.append(state_i)
            weights.append(n_i)

        global_state = average_states(client_states, weights)
        set_state(model, global_state)

        v_mae, v_rmse = evaluate(model, eval_valid, device)
        secs = time.time() - t0
        print(f"Round {r:02d} | val_MAE {v_mae:.4f} | val_RMSE {v_rmse:.4f} | {secs:.1f}s")

    set_state(model, global_state)
    te_mae, te_rmse = evaluate(model, eval_test, device)

    with open(os.path.join(OUT, "results.txt"), "w") as f:
        f.write(f"TEST MAE: {te_mae:.6f}\nTEST RMSE: {te_rmse:.6f}\n")
    torch.save(global_state, os.path.join(OUT, "fedavg_state.pt"))
    print(f"Saved {OUT}/results.txt and fedavg_state.pt")

if __name__ == "__main__":
    run()
