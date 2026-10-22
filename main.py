"""Jalankan 3 skenario pengujian untuk DTW dan Gaussian, lalu hitung akurasi.

Accuracy = (jumlah prediksi benar / jumlah seluruh data uji) x 100%
"""

from dtw import predict_dtw, train_dtw
from features import extract_mfcc
from gaussian import predict_gaussian, train_gaussian
from preprocessing import VOWELS, load_dataset

SCENARIOS = {
    "Closed Recordings": "data/template",
    "Closed Speakers": "data/test/closed",
    "Open Speakers": "data/test/open",
}


def load_features(folder):
    data = load_dataset(folder)
    for d in data:
        d["feat"] = extract_mfcc(d["signal"], d["sr"])
    return data


def accuracy(results):
    """results: list of (label, prediction)"""
    if not results:
        return 0.0
    correct = sum(1 for label, pred in results if label == pred)
    return correct / len(results) * 100


def evaluate(test_data, predict_fn, model):
    """Return list of {"speaker", "vowel", "pred"}"""
    # TODO: prediksi tiap data uji dengan predict_fn(d["feat"], model)
    raise NotImplementedError


def report(name, outputs):
    """Cetak akurasi total, per vowel (tabel 1), dan per speaker (tabel 2 & 3)."""
    # TODO
    raise NotImplementedError


def main():
    train_data = load_features("data/template")
    templates = train_dtw(train_data)
    models = train_gaussian(train_data)

    for scenario, folder in SCENARIOS.items():
        test_data = load_features(folder)
        report(f"DTW - {scenario}", evaluate(test_data, predict_dtw, templates))
        report(f"Gaussian - {scenario}", evaluate(test_data, predict_gaussian, models))


if __name__ == "__main__":
    main()
