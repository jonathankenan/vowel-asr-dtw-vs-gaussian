"""Pipeline eksperimen, prediksi, dan metrik evaluasi vowel."""

from collections import Counter
from pathlib import Path

import numpy as np

from dtw import predict_dtw, train_dtw
from features import extract_mfcc
from gaussian import predict_gaussian, train_gaussian
from preprocessing import TARGET_SR, VOWELS, load_records

DEFAULT_SCENARIOS = {
    "Closed Recording": Path("data/template"),
    "Closed Speakers": Path("data/test/closed"),
    "Open Speakers": Path("data/test/open"),
}
SYSTEMS = ("DTW", "Gaussian")
PREDICTORS = {"DTW": predict_dtw, "Gaussian": predict_gaussian}


def find_project_root(start=None, project_root=None):
    """Temukan root proyek dari konfigurasi atau direktori kerja saat ini."""
    if project_root is not None:
        root = Path(project_root).expanduser().resolve()
        if not (root / "data" / "template").is_dir():
            raise FileNotFoundError(f"Folder data/template tidak ditemukan di {root}")
        return root
    start = Path(start or Path.cwd()).expanduser().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "data" / "template").is_dir() and (candidate / "evaluation.py").is_file():
            return candidate
    raise FileNotFoundError("Root proyek tidak ditemukan. Isi PROJECT_ROOT pada sel konfigurasi.")


def load_features(records, target_sr=TARGET_SR, cache=None):
    """Tambahkan fitur MFCC ke record, menggunakan cache berbasis path opsional."""
    cache = cache if cache is not None else {}
    prepared = []
    for record in records:
        path = record["path"].resolve()
        if path not in cache:
            cache[path] = extract_mfcc(record["signal"], record["sr"], target_sr)
        prepared.append({**record, "feat": cache[path]})
    return prepared


def run_experiment(root=".", scenarios=None, target_sr=TARGET_SR):
    """Muat dataset, latih kedua sistem, dan evaluasi semua skenario.

    ``root`` adalah direktori proyek. ``scenarios`` dapat berupa mapping
    ``{nama: path}``; path relatif ditafsirkan terhadap root proyek.
    Hasil mencakup data mentah/berfitur, model, prediksi, dan konfigurasi.
    """
    root = Path(root).expanduser().resolve()
    scenario_paths = scenarios or DEFAULT_SCENARIOS
    scenario_paths = {
        name: (Path(path) if Path(path).is_absolute() else root / path)
        for name, path in scenario_paths.items()
    }
    raw = {
        name: load_records(name, folder)
        for name, folder in scenario_paths.items()
    }

    if "Closed Recording" not in raw:
        raise ValueError("Skenario data latih harus bernama 'Closed Recording'.")
    train_raw = raw["Closed Recording"]
    if not train_raw:
        raise FileNotFoundError(f"Tidak ada WAV data latih di {scenario_paths['Closed Recording']}")
    missing = set(VOWELS) - {record["vowel"] for record in train_raw}
    if missing:
        raise ValueError(f"Data template tidak memiliki kelas vowel: {sorted(missing)}")

    feature_cache = {}
    test = {
        name: load_features(records, target_sr, feature_cache)
        for name, records in raw.items()
    }
    train = test["Closed Recording"]
    models = {
        "DTW": train_dtw(train),
        "Gaussian": train_gaussian(train),
    }

    results = {}
    for system, predict in PREDICTORS.items():
        for scenario, records in test.items():
            results[(system, scenario)] = [
                {
                    "speaker": record["speaker"],
                    "speaker_group": record["speaker_group"],
                    "vowel": record["vowel"],
                    "pred": predict(record["feat"], models[system]),
                    "path": record["path"],
                }
                for record in records
            ]

    return {
        "root": root,
        "scenarios": scenario_paths,
        "raw": raw,
        "train": train,
        "test": test,
        "models": models,
        "systems": SYSTEMS,
        "results": results,
    }


def accuracy(rows):
    """Persentase prediksi benar; NaN jika tidak ada data."""
    if not rows:
        return float("nan")
    return 100.0 * sum(row["vowel"] == row["pred"] for row in rows) / len(rows)


def summary_rows(results, systems=SYSTEMS, scenarios=None):
    """Data tabel akurasi per sistem dan macro-average skenario."""
    scenarios = list(scenarios or dict.fromkeys(scenario for _, scenario in results))
    rows = []
    for system in systems:
        values = [accuracy(results.get((system, scenario), [])) for scenario in scenarios]
        available = [value for value in values if np.isfinite(value)]
        rows.append([system, *values, float(np.mean(available)) if available else float("nan")])
    return scenarios, rows


def per_vowel_rows(results, systems=SYSTEMS, scenarios=None, vowels=VOWELS):
    """Data tabel akurasi untuk setiap label vowel aktual."""
    scenarios = list(scenarios or dict.fromkeys(scenario for _, scenario in results))
    columns = [(system, scenario) for system in systems for scenario in scenarios]
    rows = []
    for vowel in vowels:
        values = [
            accuracy([r for r in results.get(column, []) if r["vowel"] == vowel])
            for column in columns
        ]
        rows.append([vowel, *values])
    rows.append(["Semua vowel", *[accuracy(results.get(column, [])) for column in columns]])
    return columns, rows


def per_speaker_rows(results, selected_scenarios, systems=SYSTEMS):
    """Data tabel per-speaker/group (macro-average per speaker)."""
    columns = [(system, scenario) for system in systems for scenario in selected_scenarios]
    groups = sorted({
        row["speaker_group"]
        for column in columns
        for row in results.get(column, [])
    })
    rows = []
    for group in groups:
        values = [
            accuracy([r for r in results.get(column, []) if r["speaker_group"] == group])
            for column in columns
        ]
        available = [value for value in values if np.isfinite(value)]
        rows.append([group, *values, float(np.mean(available)) if available else float("nan")])
    if rows:
        averages = [
            float(np.nanmean([row[index] for row in rows]))
            if any(np.isfinite(row[index]) for row in rows) else float("nan")
            for index in range(1, len(rows[0]))
        ]
        rows.append(["Rata-rata speaker", *averages])
    return columns, rows


def confusion_matrix(rows, vowels=VOWELS):
    """Matriks confusion (baris aktual, kolom prediksi) dalam urutan ``vowels``."""
    indices = {vowel: index for index, vowel in enumerate(vowels)}
    matrix = np.zeros((len(vowels), len(vowels)), dtype=int)
    for row in rows:
        if row["vowel"] in indices and row["pred"] in indices:
            matrix[indices[row["vowel"]], indices[row["pred"]]] += 1
    return matrix


def dataset_stats(records):
    """Ringkasan jumlah WAV, speaker group, vowel, sample rate, kanal, dan tipe."""
    return {
        "count": len(records),
        "groups": len({record["speaker_group"] for record in records}),
        "vowels": dict(sorted(Counter(record["vowel"] for record in records).items())),
        "sample_rates": sorted({record["sr"] for record in records}),
        "channels": sorted({1 if record["signal"].ndim == 1 else record["signal"].shape[1] for record in records}),
        "dtypes": sorted({str(record["signal"].dtype) for record in records}),
    }
