"""Load rekaman wav dan ambil label dari nama file.

Format nama file: <speaker>_<vowel>.wav, contoh S01_a.wav
"""

import os

from scipy.io import wavfile

VOWELS = ["a", "i", "u", "e", "o"]


def load_wav(path):
    """Baca file wav, kembalikan (signal, sample_rate)."""
    sr, signal = wavfile.read(path)
    # TODO: jika stereo, ambil satu channel saja
    return signal, sr


def parse_filename(filename):
    """S01_a.wav -> ("S01", "a")"""
    name = os.path.splitext(filename)[0]
    speaker, vowel = name.split("_")
    return speaker, vowel


def load_dataset(folder):
    """Baca semua wav di folder.

    Return: list of dict {"signal", "sr", "speaker", "vowel"}
    """
    data = []
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".wav"):
            continue
        signal, sr = load_wav(os.path.join(folder, filename))
        speaker, vowel = parse_filename(filename)
        data.append({"signal": signal, "sr": sr, "speaker": speaker, "vowel": vowel})
    return data
