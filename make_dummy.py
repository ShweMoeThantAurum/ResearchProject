import pickle

dummy = {
    "w": [1.0, -2.0, 0.5, 3.0, -0.1],
    "b": [0.3, -0.7, 1.2]
}

with open("dummy_update.pkl", "wb") as f:
    pickle.dump(dummy, f)
