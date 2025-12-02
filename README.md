# Adaptive Energy-Aware Federated Learning (AEFL)

This project implements a cloud-based federated learning framework for
energy-efficient, privacy-preserving traffic flow prediction using
lightweight GRU models.

The system includes:
- Multiple client roles (roadside, vehicle, sensor, camera, bus)
- AEFL adaptive client selection
- Differential privacy noise injection
- Optional local model compression
- S3-based model exchange
- Full experiment logging and summary generation

To run:

```bash
export DATASET=sz   # or los, pems08
docker compose up --build
DATASET=sz FL_MODE=AEFL python -u -m src.fl.server
