"""Load rekaman wav dan ambil label dari nama file.

Format nama file: <speaker>_<vowel>.wav, contoh S01_a.wav
"""

import os

from scipy.io import wavfile

VOWELS = ["a", "i", "u", "e", "o"]


def load_wav(path):
    """Baca file wav (mono, 16 kHz, 16-bit), kembalikan (signal, sample_rate)."""
    sr, signal = wavfile.read(path)
    return signal, sr


def parse_filename(filename):
    """S01_a.wav / S01_A.wav -> ("S01", "a")"""
    name = os.path.splitext(filename)[0]
    speaker, vowel = name.split("_")
    return speaker, vowel.lower()


def load_dataset(folder):
    """Baca semua wav di folder, termasuk subfolder per anggota.

    Return: list of dict {"signal", "sr", "speaker", "vowel"}
    """
    data = []
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for filename in sorted(files):
            if not filename.endswith(".wav"):
                continue
            signal, sr = load_wav(os.path.join(root, filename))
            speaker, vowel = parse_filename(filename)
            data.append({"signal": signal, "sr": sr, "speaker": speaker, "vowel": vowel})
    return data
