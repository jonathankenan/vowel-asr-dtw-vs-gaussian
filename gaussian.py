"""Sistem 2: ASR berbasis Model Gaussian.

Tiap vowel dimodelkan dengan satu Gaussian (mean + full covariance).
Prediksi = vowel dengan negative log-likelihood terkecil.
"""

import numpy as np
from scipy.linalg import cholesky, solve_triangular

# Regularisasi covariance: cov + REG * diag(var). Diperlukan karena jumlah frame
# per vowel bisa < 39 (rekaman ~0.35 s = ~34 frame), sehingga cov singular.
REG = 1e-2
# Batas bawah varians agar dimensi yang konstan tidak membuat cov singular
VAR_FLOOR = 1e-6


def fit_gaussian(feats, reg=REG):
    """feats: list of array (T_i, 39) dari satu vowel.

    Gabungkan semua frame, lalu hitung mean (39,) dan covariance (39, 39).
    Return: (mean, cov)
    """
    X = np.vstack(feats)
    mean = X.mean(axis=0)
    diff = X - mean
    cov = diff.T @ diff / len(X)  # maximum likelihood (bagi N)

    # tambahkan nilai kecil di diagonal agar cov positive definite dan bisa di-invers;
    # sebanding dengan varians tiap dimensi karena skala c0 jauh lebih besar dari delta
    var = np.maximum(np.diag(cov), VAR_FLOOR)
    cov = cov + np.diag(reg * var + VAR_FLOOR)
    return mean, cov


def nll(feat, mean, cov):
    """Negative log-likelihood feat (T, 39) terhadap Gaussian (mean, cov).

    -log N(x) = 0.5 * [d*log(2*pi) + log|cov| + (x-mean)^T cov^-1 (x-mean)]
    Dirata-rata untuk semua frame agar tidak bergantung pada durasi rekaman.
    """
    d = len(mean)
    # cov = L L^T  ->  log|cov| = 2 * sum(log diag L),  (x-m)^T cov^-1 (x-m) = ||L^-1 (x-m)||^2
    L = cholesky(cov, lower=True)
    log_det = 2 * np.sum(np.log(np.diag(L)))
    z = solve_triangular(L, (feat - mean).T, lower=True)
    mahalanobis = np.sum(z**2, axis=0)  # (T,)
    frame_nll = 0.5 * (d * np.log(2 * np.pi) + log_det + mahalanobis)
    return frame_nll.mean()


def train_gaussian(train_data, reg=REG):
    """train_data: list of {"feat", "vowel", ...}

    Return: dict {vowel: (mean, cov)}
    """
    feats_by_vowel = {}
    for d in train_data:
        feats_by_vowel.setdefault(d["vowel"], []).append(d["feat"])
    return {vowel: fit_gaussian(feats, reg) for vowel, feats in feats_by_vowel.items()}


def score_gaussian(feat, models):
    """Return dict {vowel: nll} untuk semua model."""
    return {vowel: nll(feat, mean, cov) for vowel, (mean, cov) in models.items()}


def predict_gaussian(feat, models):
    """Return vowel dengan nll terkecil."""
    scores = score_gaussian(feat, models)
    return min(scores, key=scores.get)
