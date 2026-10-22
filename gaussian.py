"""Sistem 2: ASR berbasis Model Gaussian.

Tiap vowel dimodelkan dengan satu Gaussian (mean + covariance).
Prediksi = vowel dengan negative log-likelihood terkecil.
"""

import numpy as np


def fit_gaussian(feats):
    """feats: list of array (T_i, 39) dari satu vowel.

    Gabungkan semua frame, lalu hitung mean (39,) dan covariance (39, 39).
    Return: (mean, cov)
    """
    # TODO: tambahkan nilai kecil di diagonal cov agar bisa di-invers
    raise NotImplementedError


def nll(feat, mean, cov):
    """Negative log-likelihood feat (T, 39) terhadap Gaussian (mean, cov).

    -log N(x) = 0.5 * [d*log(2*pi) + log|cov| + (x-mean)^T cov^-1 (x-mean)]
    Dijumlah atau dirata-rata untuk semua frame.
    """
    raise NotImplementedError


def train_gaussian(train_data):
    """train_data: list of {"feat", "vowel", ...}

    Return: dict {vowel: (mean, cov)}
    """
    raise NotImplementedError


def predict_gaussian(feat, models):
    """Return vowel dengan nll terkecil."""
    raise NotImplementedError
