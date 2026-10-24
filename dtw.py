"""Sistem 1: Template-based ASR dengan DTW (averaged template).

Input -> MFCC -> DTW terhadap tiap template -> pilih cost terkecil -> vowel
"""

import numpy as np
from scipy.spatial.distance import cdist


def dtw(X, Y):
    """Hitung cost DTW antara X (n, 39) dan Y (m, 39).

    Jarak lokal: Euclidean antar frame.
    Langkah yang diizinkan: (i-1, j), (i, j-1), (i-1, j-1).
    Cost dinormalisasi dengan panjang path agar bisa dibandingkan antar template
    yang panjangnya berbeda.
    Return: (cost, path), path = list pasangan index (i, j) yang ter-align
    """
    n, m = len(X), len(Y)
    dist = cdist(X, Y, metric="euclidean")

    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = dist[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        steps = {(i - 1, j - 1): D[i - 1, j - 1], (i - 1, j): D[i - 1, j], (i, j - 1): D[i, j - 1]}
        i, j = min(steps, key=steps.get)
    path.reverse()

    return D[n, m] / len(path), path


def build_avg_template(feats):
    """Buat averaged template dari beberapa rekaman satu vowel (1 per speaker).

    1. Pilih satu rekaman sebagai referensi
    2. Align rekaman lain ke referensi dengan dtw()
    3. Rata-ratakan frame yang ter-align ke tiap frame referensi
    Return: array (T_ref, 39)
    """
    order = np.argsort([len(f) for f in feats])
    ref = feats[order[len(order) // 2]]

    total = np.zeros_like(ref, dtype=float)
    for feat in feats:
        _, path = dtw(ref, feat)
        aligned_sum = np.zeros_like(ref, dtype=float)
        aligned_count = np.zeros(len(ref))
        for i, j in path:
            aligned_sum[i] += feat[j]
            aligned_count[i] += 1
        total += aligned_sum / aligned_count[:, None]
    return total / len(feats)


def train_dtw(train_data):
    """train_data: list of {"feat", "vowel", ...}

    Return: dict {vowel: averaged template}
    """
    feats_by_vowel = {}
    for d in train_data:
        feats_by_vowel.setdefault(d["vowel"], []).append(d["feat"])
    return {vowel: build_avg_template(feats) for vowel, feats in feats_by_vowel.items()}


def score_dtw(feat, templates):
    """Return dict {vowel: cost DTW} untuk semua template."""
    return {vowel: dtw(feat, template)[0] for vowel, template in templates.items()}


def predict_dtw(feat, templates):
    scores = score_dtw(feat, templates)
    return min(scores, key=scores.get)
