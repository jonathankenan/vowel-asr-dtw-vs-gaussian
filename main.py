"""Jalankan eksperimen pengenalan vowel tanpa notebook.

Contoh: ``python main.py`` atau ``python main.py --root C:/proyek``.
"""

import argparse
from pathlib import Path

from evaluation import run_experiment
from reporting import print_console_report, save_predictions


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluasi ASR vowel DTW dan Gaussian")
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parent,
        help="Direktori proyek yang berisi folder data/ (default: direktori main.py)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Lokasi CSV prediksi (default: <root>/outputs/predictions.csv)",
    )
    parser.add_argument("--no-csv", action="store_true", help="Jangan ekspor prediksi ke CSV")
    args = parser.parse_args(argv)

    experiment = run_experiment(args.root)
    print_console_report(experiment)
    if not args.no_csv:
        output_path = save_predictions(experiment["results"], experiment["root"], args.output)
        total = sum(map(len, experiment["results"].values()))
        print(f"\nPrediksi disimpan: {output_path} ({total} baris)")
    return experiment


if __name__ == "__main__":
    main()
