"""Sistem 1: Template-based ASR dengan DTW (averaged template).

Input -> MFCC -> DTW terhadap tiap template -> pilih cost terkecil -> vowel
"""

import numpy as np


def dtw(X, Y):
    """Hitung cost DTW antara X (n, 39) dan Y (m, 39).

    Jarak lokal: Euclidean antar frame.
    Return: (cost, path), path = list pasangan index (i, j) yang ter-align
    """
    # TODO: isi matriks cost dengan dynamic programming
    # TODO: backtrack untuk mendapatkan path
    raise NotImplementedError


def build_avg_template(feats):
    """Buat averaged template dari beberapa rekaman satu vowel (1 per speaker).

    1. Pilih satu rekaman sebagai referensi
    2. Align rekaman lain ke referensi dengan dtw()
    3. Rata-ratakan frame yang ter-align ke tiap frame referensi
    Return: array (T_ref, 39)
    """
    raise NotImplementedError


def train_dtw(train_data):
    """train_data: list of {"feat", "vowel", ...}

    Return: dict {vowel: averaged template}
    """
    raise NotImplementedError


def predict_dtw(feat, templates):
    """Return vowel dengan cost DTW terkecil."""
    raise NotImplementedError
