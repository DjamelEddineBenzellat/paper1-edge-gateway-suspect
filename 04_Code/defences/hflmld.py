"""HFLMLD defences for P-PE (PR §5, DL-37), from Bi et al., HFLMLD, IEEE IoT-J 12(24), 2025, §IV.

Edge (§IV-A, IV-B, Algorithm 1 lines 12-18):
  min-max normalization of each client's model parameters (eq. 4), PCA (eq. 5), K-means
  -> detection V (1 = malicious); suspicion score S += 2V - 1 over the edge iterations of one
  cloud round (eq. 6-7), reset at each cloud round; a client enters the edge mean if it is
  benign now and S <= 0; the edge model is the plain mean of those clients' models (line 18).
Cloud (§IV-C, eq. 8-14):
  g_e = (w* - w_e) / eta; log normalization g_e <- ln(1 + |g_e|) g_e / |g_e|;
  CR_e = ReLU(cos(w_e, w*) * (1 - min-max-normalized |w_e - w*|));
  w* <- w* - eta * sum(CR_e g_e) / sum(CR_e).
Choices the paper does not state are marked "[choice]" and logged in
05_Experiments/hflmld_differences.md.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

PCA_COMPONENTS = 2  # [choice] the paper does not give the PCA dimension
KMEANS_CLUSTERS = 2  # [choice] benign / malicious


def minmax_per_client(models):
    """Eq. (4), [choice] per client: scale each client's parameter vector to [0, 1]."""
    lo = models.min(axis=1, keepdims=True)
    span = models.max(axis=1, keepdims=True) - lo
    return (models - lo) / np.where(span > 0, span, 1.0)


def edge_detect(models, seed):
    """Detection V (1 = malicious) for the client models (rows), and diagnostics.

    [choice] The smaller K-means cluster is malicious; with two equal clusters nobody is flagged.
    """
    n = len(models)
    if n < 3:
        return np.zeros(n, dtype=int), {"cluster_sizes": [n], "flagged": 0}
    reduced = PCA(n_components=min(PCA_COMPONENTS, n - 1), random_state=seed).fit_transform(minmax_per_client(models))
    labels = KMeans(n_clusters=KMEANS_CLUSTERS, n_init=10, random_state=seed).fit_predict(reduced)
    sizes = np.bincount(labels, minlength=KMEANS_CLUSTERS)
    if sizes[0] == sizes[1]:
        v = np.zeros(n, dtype=int)
    else:
        v = (labels == int(np.argmin(sizes))).astype(int)
    return v, {"cluster_sizes": sorted(int(s) for s in sizes), "flagged": int(v.sum())}


def update_suspicion(scores, v):
    """Eq. (6): S <- S + 2V - 1."""
    return scores + 2 * v - 1


def admitted(scores, v):
    """§IV-B: in the edge mean if benign in this edge iteration and suspicion score <= 0."""
    return (v == 0) & (scores <= 0)


def cloud_aggregate(global_vec, edge_models, eta):
    """Eq. (8)-(14). edge_models: {gateway: model vector}. Returns the new global model and diagnostics."""
    gws = sorted(edge_models)
    w_star = np.asarray(global_vec, dtype=np.float64)
    w = np.stack([np.asarray(edge_models[g], dtype=np.float64) for g in gws])
    g = (w_star[None, :] - w) / eta
    norms = np.linalg.norm(g, axis=1)
    g_hat = np.where(norms[:, None] > 0, np.log1p(norms)[:, None] * g / np.where(norms > 0, norms, 1.0)[:, None], 0.0)
    cs = (w @ w_star) / (np.linalg.norm(w, axis=1) * np.linalg.norm(w_star) + 1e-12)
    euc = np.linalg.norm(w - w_star[None, :], axis=1)
    span = euc.max() - euc.min()
    euc_n = (euc - euc.min()) / span if span > 0 else np.zeros_like(euc)  # [choice] equal distances -> 0
    cr = np.maximum(cs * (1.0 - euc_n), 0.0)
    if cr.sum() > 0:
        new = w_star - eta * (cr[:, None] * g_hat).sum(axis=0) / cr.sum()
    else:  # [choice] every weight is 0: keep the global model
        new = w_star
    info = {"weights": {int(gw): float(c) for gw, c in zip(gws, cr)}, "cosine": [float(c) for c in cs],
            "distance": [float(e) for e in euc], "step": float(np.linalg.norm(new - w_star))}  # DL-40
    return new.astype(np.float32), info
