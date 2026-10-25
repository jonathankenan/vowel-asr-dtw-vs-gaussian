"""Pembacaan, pelabelan, dan standardisasi rekaman WAV."""

import re
from pathlib import Path
from math import gcd

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

VOWELS = ("a", "i", "u", "e", "o")
TARGET_SR = 16000
VOWEL_SUFFIX = re.compile(r"(?:^|[_-])([aiueo])(?:\d+)?$", re.IGNORECASE)


def load_wav(path):
    """Baca WAV dan kembalikan ``(signal, sample_rate)``."""
    sr, signal = wavfile.read(path)
    return signal, int(sr)


def vowel_from_filename(path):
    """Ambil vowel dari suffix nama file, misalnya ``G09_S01_A01.wav``."""
    match = VOWEL_SUFFIX.search(Path(path).stem)
    return match.group(1).lower() if match else None


def parse_filename(filename):
    """Parser sederhana kompatibel: ``S01_A.wav`` menjadi ``("S01", "a")``."""
    vowel = vowel_from_filename(filename)
    if vowel is None:
        raise ValueError(f"Label vowel tidak ditemukan pada nama file: {filename}")
    speaker = speaker_from_filename(filename)
    return speaker, vowel


def speaker_from_filename(path):
    """Ambil identitas speaker dari nama file dengan menghapus suffix vowel."""
    stem = Path(path).stem
    return VOWEL_SUFFIX.sub("", stem).rstrip("_- ") or stem


def vowel_from_relative_folders(path, base, vowels=VOWELS):
    """Cari komponen folder vowel terdekat, misalnya ``open/A/file.wav``."""
    path, base = Path(path), Path(base)
    for component in reversed(path.relative_to(base).parts[:-1]):
        if component.lower() in vowels:
            return component.lower()
    return None


def prepare_audio(signal, sr, target_sr=TARGET_SR):
    """Ubah audio ke mono float dan sample rate target sebelum ekstraksi fitur."""
    signal = np.asarray(signal)
    if signal.ndim == 2:
        signal = signal.astype(np.float64).mean(axis=1)
    elif signal.ndim == 1:
        signal = signal.astype(np.float64)
    else:
        raise ValueError(f"Bentuk audio tidak didukung: {signal.shape}")
    if not np.all(np.isfinite(signal)):
        raise ValueError("Sinyal audio mengandung NaN/Inf")
    if sr <= 0:
        raise ValueError(f"Sample rate tidak valid: {sr}")
    if target_sr <= 0:
        raise ValueError(f"Target sample rate tidak valid: {target_sr}")
    if sr != target_sr:
        divisor = gcd(int(sr), int(target_sr))
        signal = resample_poly(signal, target_sr // divisor, int(sr) // divisor)
    min_samples = int(0.025 * target_sr)
    if len(signal) < min_samples:
        raise ValueError(f"Audio terlalu pendek untuk satu frame MFCC ({len(signal)} sampel)")
    return signal


def load_records(name, folder, vowels=VOWELS):
    """Muat semua WAV rekursif dan tentukan label serta identitas speaker.

    Untuk ``Open Speakers``, folder vowel menjadi label ground truth utama.
    Untuk template/closed, folder tingkat pertama menjadi identitas speaker.
    Record menyertakan ``path`` agar fitur dapat dicache lintas skenario.
    """
    folder = Path(folder)
    records = []
    if not folder.is_dir():
        return records

    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".wav":
            continue
        signal, sr = load_wav(path)
        filename_vowel = vowel_from_filename(path)
        folder_vowel = vowel_from_relative_folders(path, folder, vowels)

        if name.lower().startswith("open"):
            vowel = folder_vowel or filename_vowel
            if folder_vowel and filename_vowel and folder_vowel != filename_vowel:
                print(f"PERINGATAN label folder/nama berbeda: {path} ({folder_vowel} vs {filename_vowel})")
            speaker = speaker_from_filename(path)
            speaker_group = speaker
        else:
            vowel = filename_vowel or folder_vowel
            relative_parts = path.relative_to(folder).parts
            speaker_group = relative_parts[0] if len(relative_parts) > 1 else speaker_from_filename(path)
            speaker = speaker_group

        if vowel not in vowels:
            raise ValueError(f"Label vowel tidak dapat ditentukan untuk: {path}")
        records.append({
            "signal": signal,
            "sr": sr,
            "speaker": speaker,
            "speaker_group": speaker_group,
            "vowel": vowel,
            "path": path,
        })
    return records


def load_dataset(folder):
    """Kompatibilitas sederhana: muat data dari satu direktori."""
    return load_records("dataset", folder)
