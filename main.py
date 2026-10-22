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
SYSTEMS = ["DTW", "Gaussian"]


def load_features(folder):
    data = load_dataset(folder)
    for d in data:
        d["feat"] = extract_mfcc(d["signal"], d["sr"])
    return data


def accuracy(outputs):
    """outputs: list of {"speaker", "vowel", "pred"}"""
    if not outputs:
        return 0.0
    correct = sum(1 for o in outputs if o["vowel"] == o["pred"])
    return correct / len(outputs) * 100


def evaluate(test_data, predict_fn, model):
    """Return list of {"speaker", "vowel", "pred"}"""
    return [
        {"speaker": d["speaker"], "vowel": d["vowel"], "pred": predict_fn(d["feat"], model)}
        for d in test_data
    ]


def filter_by(outputs, key, value):
    return [o for o in outputs if o[key] == value]


def speakers_of(outputs):
    return sorted({o["speaker"] for o in outputs})


def mean(values):
    return sum(values) / len(values) if values else 0.0


def print_table(title, header, rows):
    print(f"\n{title}")
    print(f"{header[0]:<18}" + "".join(f"{h:>28}" for h in header[1:]))
    for row in rows:
        print(f"{row[0]:<18}" + "".join(f"{v:>27.2f}%" for v in row[1:]))


def print_summary(results):
    """Akurasi tiap sistem per skenario + rata-rata."""
    header = ["Sistem"] + list(SCENARIOS) + ["Rata-rata"]
    rows = []
    for system in SYSTEMS:
        accs = [accuracy(results[system, s]) for s in SCENARIOS]
        rows.append([system] + accs + [mean(accs)])
    print_table("Ringkasan Akurasi", header, rows)


def print_per_vowel(results):
    """Tabel 1: akurasi untuk setiap vowel."""
    columns = [(system, s) for system in SYSTEMS for s in SCENARIOS]
    header = ["Vowel"] + [f"{system} {s}" for system, s in columns]
    rows = []
    for vowel in VOWELS:
        rows.append([vowel] + [accuracy(filter_by(results[c], "vowel", vowel)) for c in columns])
    rows.append(["Average Accuracy"] + [accuracy(results[c]) for c in columns])
    print_table("1. Accuracy untuk Setiap Vowel", header, rows)


def print_per_speaker(title, results, scenarios):
    """Tabel 2 & 3: akurasi per speaker."""
    columns = [(system, s) for system in SYSTEMS for s in scenarios]
    header = ["Speakers"] + [f"{system} {s}" for system, s in columns] + ["Average"]
    rows = []
    for speaker in speakers_of(results[columns[0]]):
        accs = [accuracy(filter_by(results[c], "speaker", speaker)) for c in columns]
        rows.append([speaker] + accs + [mean(accs)])
    if rows:
        col_avgs = [mean([row[i] for row in rows]) for i in range(1, len(header))]
        rows.append(["Average"] + col_avgs)
    print_table(title, header, rows)


def main():
    train_data = load_features("data/template")
    models = {"DTW": train_dtw(train_data), "Gaussian": train_gaussian(train_data)}
    predict_fns = {"DTW": predict_dtw, "Gaussian": predict_gaussian}

    results = {}
    for scenario, folder in SCENARIOS.items():
        test_data = load_features(folder)
        for system in SYSTEMS:
            results[system, scenario] = evaluate(test_data, predict_fns[system], models[system])

    print_summary(results)
    print_per_vowel(results)
    print_per_speaker("2. Closed Experiments", results, ["Closed Recordings", "Closed Speakers"])
    print_per_speaker("3. Open Experiments", results, ["Open Speakers"])


if __name__ == "__main__":
    main()
