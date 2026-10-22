"""Ekstraksi MFCC 39 dimensi per frame.

39 = 13 MFCC (koefisien ke-0 diganti log energy) + 13 delta + 13 delta-delta
"""

import numpy as np
from python_speech_features import delta, mfcc


def extract_mfcc(signal, sr):
    """Return array (T, 39), T = jumlah frame."""
    # TODO: mfcc(signal, sr, winlen=0.025, winstep=0.01, numcep=13, appendEnergy=True)
    # TODO: d1 = delta(mfcc, 2), d2 = delta(d1, 2)
    # TODO: gabungkan dengan np.hstack -> (T, 39)
    raise NotImplementedError
