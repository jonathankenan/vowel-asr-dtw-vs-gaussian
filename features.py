"""Ekstraksi MFCC 39 dimensi per frame.

39 = 13 MFCC (koefisien ke-0 diganti log energy) + 13 delta + 13 delta-delta
"""

import numpy as np
from python_speech_features import delta, mfcc
from preprocessing import TARGET_SR, prepare_audio


def extract_mfcc(signal, sr, target_sr=TARGET_SR):
    """Return array (T, 39), T = jumlah frame."""
    signal = prepare_audio(signal, sr, target_sr)
    # frame 25 ms, geser 10 ms
    static = mfcc(signal, target_sr, winlen=0.025, winstep=0.01, numcep=13, appendEnergy=True)
    d1 = delta(static, 2)
    d2 = delta(d1, 2)
    return np.hstack([static, d1, d2])
