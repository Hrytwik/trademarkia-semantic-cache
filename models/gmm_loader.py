# models/gmm_loader.py
import joblib
import numpy as np
from sklearn.mixture import GaussianMixture

def load_gmm() -> GaussianMixture:
    return joblib.load("artifacts/gmm_diag20_model.pkl")

def get_query_cluster_probs(gmm: GaussianMixture, query_embedding: np.ndarray) -> np.ndarray:
    return gmm.predict_proba(query_embedding.reshape(1, -1))[0]
