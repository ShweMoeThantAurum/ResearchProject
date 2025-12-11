# Adaptive Energy-Aware Federated Learning (AEFL)

A modular federated learning framework designed for running FL experiments with **adaptive client selection**, **energy estimation**, **differential privacy**, and **optional compression**, using **Docker-based clients** and an **S3-compatible storage backend** for communication.

This repo includes:

* Federated server implementation
* Dockerised IoT-client simulation
* GRU-based forecasting model
* Energy logging & metadata tracking
* Differential privacy noise injection
* Experiment scripts & plotting utilities

---

## 🔧 Features

* **Federated Learning Modes**

  * FedAvg
  * FedProx
  * AEFL (adaptive energy-aware scheduling)

* **Client Simulation**

  * Each client runs in its own Docker container
  * Non-IID data partitions
  * S3-style coordination (e.g., MinIO, AWS S3)

* **Energy & Metadata Tracking**

  * Training time
  * Upload/download size
  * Approx. energy per round

* **Optional Modules**

  * Differential privacy (Gaussian noise)
  * Model compression (pruning, top-k, quantisation)

* **Plotting Tools**

  * Accuracy curves
  * Energy comparison
  * Privacy–accuracy/energy trade-off

---

## 📁 Project Structure

```
src/
 ├── fl/
 │    ├── server/        # Client selection, aggregation, evaluation
 │    └── client/        # Local training loop, DP, compression, energy logs
 ├── models/             # GRU model
 ├── utils/              # Helper functions (DP, compression, IO)
data/                    # Preprocessed dataset partitions
outputs/                 # Metrics, logs, plots
scripts/                 # Experiment automation
docker/                  # Docker configs for clients
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start S3-compatible backend (e.g., MinIO)

```bash
docker-compose up -d
```

### 3. Launch the federated server

```bash
python src/fl/server/run_server.py --mode aefl --dataset pems08
```

### 4. Start client containers

```bash
bash scripts/run_clients.sh
```

### 5. Run differential privacy experiments

```bash
bash scripts/run_dp_experiments.sh
```

Results (metrics + plots) are saved under:

```
outputs/<dataset>/<mode>/
```

---

## 📊 Example Outputs

* Energy across FL modes
* Accuracy (MAE/RMSE/MAPE)
* DP vs accuracy
* DP vs energy

Plots are automatically generated and saved in `outputs/`.

---

## 🧪 Supported Datasets

* `Los-Loop`
* `PeMSD8`
* `SZ-Taxi`

All datasets use the same preprocessing pipeline (normalisation, alignment, sliding-window, non-IID partitioning).
