"""Utility metrics (CLAUDE.md metrics; DL-25, DL-33, DL-35, DL-36).

DL-36 (deciding metric from 15 Nov 2026): macro-F1 and per-class recall at a
per-class threshold tuned on fold 9 (grid 0.01-0.99, best F1, ties: the middle
one; a class without positive records keeps 0.5). The same thresholds are
applied to fold 10. Keys: macro_f1_tuned, per_class_f1_tuned,
per_class_recall_tuned, thresholds.
DL-33 (threshold 0.5) keeps its old keys macro_f1, per_class_f1 and
per_class_recall, so the G0-a result files keep their meaning; it is reported
in an appendix only. AUROC (macro and per class) is reported next to both.
"""
import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score, roc_auc_score

THRESHOLD = 0.5  # DL-33 (appendix); also the DL-36 default for a class without positive records
CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
THRESHOLD_GRID = np.round(np.linspace(0.01, 0.99, 99), 2)  # DL-36


@torch.no_grad()
def predict_proba(model, x, device, batch=512):
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        xb = torch.as_tensor(np.array(x[i:i + batch], dtype=np.float32), device=device)  # copy: x may be read-only
        out.append(torch.sigmoid(model(xb)).float().cpu())
    model.train()
    return torch.cat(out).numpy()


def tuned_thresholds(y_true, proba):
    """DL-36: per class, the threshold on THRESHOLD_GRID with the best F1 on (y_true, proba).

    Ties: the middle one of the best thresholds. A class without positive records keeps 0.5.
    """
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(proba)
    out = []
    for c in range(y.shape[1]):
        if not y[:, c].any():
            out.append(THRESHOLD)
            continue
        pred = p[None, :, c] >= THRESHOLD_GRID[:, None]
        tp = (pred & y[:, c]).sum(axis=1)
        fp = (pred & ~y[:, c]).sum(axis=1)
        fn = (~pred & y[:, c]).sum(axis=1)
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        best = np.flatnonzero(f1 == f1.max())
        out.append(float(THRESHOLD_GRID[best[len(best) // 2]]))
    return np.array(out)


def macro_f1_at(y_true, proba, thresholds):
    y_pred = (np.asarray(proba) >= np.asarray(thresholds)[None, :]).astype(int)
    return float(f1_score(np.asarray(y_true).astype(int), y_pred, average="macro", zero_division=0))


def per_class_auroc(y_true, proba):
    y_true = np.asarray(y_true).astype(int)
    out = {}
    for i, c in enumerate(CLASSES):
        try:
            out[c] = float(roc_auc_score(y_true[:, i], proba[:, i]))
        except ValueError:  # one label value only
            out[c] = None
    return out


def scores(y_true, proba, thresholds=None):
    """Scores of one model on one fold.

    thresholds=None: the DL-36 thresholds are tuned on (y_true, proba) itself (use on fold 9).
    On fold 10, pass the fold-9 thresholds.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (proba >= THRESHOLD).astype(int)
    res = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": dict(zip(CLASSES, map(float, f1_score(y_true, y_pred, average=None, zero_division=0)))),
        "per_class_recall": dict(zip(CLASSES, map(float, recall_score(y_true, y_pred, average=None,
                                                                       zero_division=0)))),
    }
    try:
        res["macro_auroc"] = float(roc_auc_score(y_true, proba, average="macro"))
    except ValueError:  # a class with one label value only
        res["macro_auroc"] = None
    t = tuned_thresholds(y_true, proba) if thresholds is None else np.asarray(thresholds, dtype=float)
    y_tuned = (proba >= t[None, :]).astype(int)
    res.update({
        "macro_f1_tuned": float(f1_score(y_true, y_tuned, average="macro", zero_division=0)),
        "per_class_f1_tuned": dict(zip(CLASSES, map(float, f1_score(y_true, y_tuned, average=None,
                                                                     zero_division=0)))),
        "per_class_recall_tuned": dict(zip(CLASSES, map(float, recall_score(y_true, y_tuned, average=None,
                                                                             zero_division=0)))),
        "thresholds": dict(zip(CLASSES, map(float, t))),
        "thresholds_tuned_on_this_fold": thresholds is None,
        "per_class_auroc": per_class_auroc(y_true, proba),
        "mi_sens_at_90spec": mi_sensitivity_at_90_specificity(y_true, proba),  # PR §3 item 31, DL-38 (3)
    })
    return res


def trivial_macro_f1(y_true):
    """DL-25: predict every class positive; per class F1 = 2p / (1 + p)."""
    p = np.asarray(y_true).mean(axis=0)
    return float(np.mean(2 * p / (1 + p)))


def diagnostic_scores(y_val, p_val, y_tune, p_tune):
    """DL-35: per-class AUROC on fold 9, and fold-9 macro-F1 with per-class thresholds tuned
    on (y_tune, p_tune) (the training data) and, as an optimistic bound, on fold 9 itself."""
    t_train = tuned_thresholds(y_tune, p_tune)
    t_fold9 = tuned_thresholds(y_val, p_val)
    return {"per_class_auroc": per_class_auroc(y_val, p_val),
            "macro_f1_train_thresholds": macro_f1_at(y_val, p_val, t_train),
            "train_thresholds": dict(zip(CLASSES, map(float, t_train))),
            "macro_f1_fold9_thresholds": macro_f1_at(y_val, p_val, t_fold9),
            "fold9_thresholds": dict(zip(CLASSES, map(float, t_fold9)))}


# ---- M2-03: the remaining PR §7 measures (definitions in DL-38) ----
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.stats import beta  # noqa: E402

MI = CLASSES.index("MI")
FS_HZ = 100
LEAD_II = 1  # PTB-XL lead order: I, II, III, AVR, AVL, AVF, V1-V6
TRIGGER = {"lead": LEAD_II, "start_s": 1.0, "length_s": 0.2, "mV": 0.5}  # PR §3 item 26


def sensitivity_at_specificity(y_true, score, specificity=0.9):
    """Largest true-positive rate with false-positive rate <= 1 - specificity, on the ROC curve."""
    from sklearn.metrics import roc_curve
    y_true = np.asarray(y_true).astype(int)
    if y_true.min() == y_true.max():
        return None
    fpr, tpr, _ = roc_curve(y_true, score)
    ok = fpr <= 1 - specificity + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def mi_sensitivity_at_90_specificity(y_true, proba):
    """PR §3 item 31 (DL-38 (3)): on the ROC curve of the MI class, on the scored fold."""
    return sensitivity_at_specificity(np.asarray(y_true)[:, MI], np.asarray(proba)[:, MI], 0.9)


def apply_trigger(x):
    """Backdoor trigger (PR §3 item 26): +0.5 mV on lead II for 0.2 s from t = 1.0 s. Returns a copy."""
    out = np.array(x, dtype=np.float32, copy=True)
    a = int(round(TRIGGER["start_s"] * FS_HZ))
    out[..., TRIGGER["lead"], a:a + int(round(TRIGGER["length_s"] * FS_HZ))] += TRIGGER["mV"]
    return out


def backdoor_success(y_true, proba_triggered, thresholds):
    """DL-38 (4): among records whose true labels include a disease class, the share labelled exactly
    'NORM only' after the trigger, at the model's DL-36 thresholds."""
    y = np.asarray(y_true).astype(bool)
    pred = np.asarray(proba_triggered) >= np.asarray(thresholds)[None, :]
    diseased = y[:, 1:].any(axis=1)
    if not diseased.any():
        return None
    norm_only = pred[:, 0] & ~pred[:, 1:].any(axis=1)
    return float(norm_only[diseased].mean())


def ecg_r(x_true, x_rec):
    """Mean Pearson r over the leads (H1: 'recognizable' if r >= 0.8, PR §3 item 32)."""
    rs = []
    for a, b in zip(np.asarray(x_true, dtype=np.float64), np.asarray(x_rec, dtype=np.float64)):
        a, b = a - a.mean(), b - b.mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        rs.append(float(a @ b) / denom if denom > 0 else 0.0)
    return float(np.mean(rs))


def _znorm(v):
    v = np.asarray(v, dtype=np.float64)
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v - v.mean()


def dtw_normalized(a, b):
    """DTW with |a_i - b_j| cost, divided by the length of the optimal warping path (one lead).
    Computed along anti-diagonals, so each step is a vector operation."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n, m = len(a), len(b)
    cost = np.full((n + 1, m + 1), np.inf)
    length = np.zeros((n + 1, m + 1))
    cost[0, 0] = 0.0
    for s in range(2, n + m + 1):  # cells (i, j) with i + j = s, 1-based
        i = np.arange(max(1, s - m), min(n, s - 1) + 1)
        j = s - i
        options = np.stack([cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1]])  # diagonal first: ties -> diagonal
        k = np.argmin(options, axis=0)
        prev_len = np.stack([length[i - 1, j - 1], length[i - 1, j], length[i, j - 1]])
        cost[i, j] = np.abs(a[i - 1] - b[j - 1]) + options[k, np.arange(len(i))]
        length[i, j] = prev_len[k, np.arange(len(i))] + 1
    return float(cost[n, m] / length[n, m])


def ecg_dtw(x_true, x_rec):
    """DL-38 (5): normalized DTW per lead on z-normalized signals; the mean over the leads."""
    return float(np.mean([dtw_normalized(_znorm(a), _znorm(b)) for a, b in zip(x_true, x_rec)]))


def psnr(img_true, img_rec, data_range=1.0):
    """Peak signal-to-noise ratio in dB (E2, MedMNIST)."""
    mse = float(np.mean((np.asarray(img_true, dtype=np.float64) - np.asarray(img_rec, dtype=np.float64)) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(data_range ** 2 / mse))


def ssim(img_true, img_rec, data_range=1.0, sigma=1.5):
    """Structural similarity of two 2-D images (Wang et al. 2004: Gaussian window sigma 1.5, K1 = 0.01,
    K2 = 0.03), the mean over the image."""
    x, y = np.asarray(img_true, dtype=np.float64), np.asarray(img_rec, dtype=np.float64)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    f = lambda v: gaussian_filter(v, sigma, mode="reflect", truncate=3.5)  # noqa: E731 (11 x 11 window)
    mx, my = f(x), f(y)
    vx, vy, cxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
    return float(s.mean())


def wilson_ci(k, n, z=1.959964):
    """Wilson 95% interval for a proportion k / n (PR §7 statistics)."""
    if n == 0:
        return (None, None)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(centre - half), float(centre + half))


def membership_auc(scores_in, scores_out):
    y = np.r_[np.ones(len(scores_in)), np.zeros(len(scores_out))]
    return float(roc_auc_score(y, np.r_[scores_in, scores_out]))


def epsilon_lower_bound(scores_in, scores_out, delta=1e-5, alpha=0.05):
    """DL-38 (2): the audit lower bound on epsilon. For every score threshold, one-sided (1 - alpha)
    Clopper-Pearson bounds give ln((TPR_lo - delta) / FPR_hi) and ln((TNR_lo - delta) / FNR_hi);
    the largest value over thresholds (0 if none is positive)."""
    s_in, s_out = np.asarray(scores_in, dtype=np.float64), np.asarray(scores_out, dtype=np.float64)
    n_in, n_out = len(s_in), len(s_out)
    lo = lambda k, n: 0.0 if k == 0 else float(beta.ppf(alpha, k, n - k + 1))  # noqa: E731
    hi = lambda k, n: 1.0 if k == n else float(beta.ppf(1 - alpha, k + 1, n - k))  # noqa: E731
    best = 0.0
    for t in np.unique(np.r_[s_in, s_out]):
        tp, fp = int((s_in >= t).sum()), int((s_out >= t).sum())
        for good, bad in ((lo(tp, n_in), hi(fp, n_out)), (lo(n_out - fp, n_out), hi(n_in - tp, n_in))):
            if good - delta > 0 and bad > 0:
                best = max(best, float(np.log((good - delta) / bad)))
    return best


def wan_bytes(mode, n_params, n_gateways, devices_per_gateway, edge_iterations=1, bytes_per_value=4):
    """WAN bytes (gateways <-> cloud) of the model vectors, per cloud round and per device round (H5(b)).
    Uplink: one vector per gateway, except P-FR (the masked message of every device) and FedAvg (every
    device's update, relayed). Downlink: the model to every gateway. 32-bit values (PR §3 item 16)."""
    if mode in ("local", "centralized"):
        return {"uplink_per_cloud_round": 0, "downlink_per_cloud_round": 0, "uplink_per_device_round": 0.0}
    vec = n_params * bytes_per_value
    senders = n_gateways * devices_per_gateway if mode in ("pfr", "fedavg") else n_gateways
    up = senders * vec
    return {"uplink_per_cloud_round": up, "downlink_per_cloud_round": n_gateways * vec,
            "uplink_per_device_round": up / edge_iterations}
