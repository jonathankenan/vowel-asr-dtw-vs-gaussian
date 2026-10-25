"""Penyajian hasil eksperimen dan ekspor prediksi."""

import csv
from pathlib import Path

import numpy as np


def format_markdown_table(headers, rows, percent=True):
    """Format tabel Markdown tanpa ketergantungan pada IPython/notebook."""
    def fmt(value):
        if isinstance(value, (int, float, np.number)):
            if not np.isfinite(value):
                return "—"
            return f"{value:.2f}%".replace(".", ",") if percent else str(int(value))
        return str(value)

    lines = ["| " + " | ".join(map(str, headers)) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(fmt(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def save_predictions(results, root, output_path=None):
    """Ekspor seluruh prediksi ke CSV; mengembalikan path file yang dibuat."""
    root = Path(root).resolve()
    output_path = Path(output_path) if output_path else root / "outputs" / "predictions.csv"
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    distinct_groups = any(
        row["speaker_group"] != row["speaker"]
        for rows in results.values() for row in rows
    )
    fieldnames = ["system", "scenario"]
    if distinct_groups:
        fieldnames.append("speaker_group")
    fieldnames.extend(["speaker", "actual", "predicted", "path"])

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for (system, scenario), rows in results.items():
            for row in rows:
                record = {
                    "system": system,
                    "scenario": scenario,
                    "speaker": row["speaker"],
                    "actual": row["vowel"],
                    "predicted": row["pred"],
                    "path": Path(row["path"]).resolve().relative_to(root).as_posix(),
                }
                if distinct_groups and row["speaker_group"] != row["speaker"]:
                    record["speaker_group"] = row["speaker_group"]
                writer.writerow(record)
    return output_path


def print_console_summary(experiment):
    """Cetak ringkasan akurasi utama untuk penggunaan CLI."""
    from evaluation import summary_rows

    scenarios, rows = summary_rows(
        experiment["results"], experiment["systems"], experiment["scenarios"]
    )
    headers = ["Sistem", *scenarios, "Rata-rata skenario"]
    widths = [max(18, len(str(header)) + 2) for header in headers]
    print("\nRingkasan Akurasi")
    print("".join(str(header).ljust(width) for header, width in zip(headers, widths)))
    for row in rows:
        print(str(row[0]).ljust(widths[0]) + "".join(
            ("—" if not np.isfinite(value) else f"{value:.2f}%").rjust(width)
            for value, width in zip(row[1:], widths[1:])
        ))


def print_console_report(experiment):
    """Cetak ringkasan, akurasi per vowel, dan akurasi per speaker."""
    from evaluation import per_speaker_rows, per_vowel_rows

    def print_table(title, headers, rows):
        widths = [max(18, *(len(str(row[i])) + 2 for row in rows), len(str(header)) + 2)
                  for i, header in enumerate(headers)] if rows else [max(18, len(str(h)) + 2) for h in headers]
        print(f"\n{title}")
        print("".join(str(header).ljust(width) for header, width in zip(headers, widths)))
        for row in rows:
            formatted = [
                str(value) if i == 0 else ("—" if not np.isfinite(value) else f"{value:.2f}%")
                for i, value in enumerate(row)
            ]
            print("".join(value.ljust(width) for value, width in zip(formatted, widths)))

    results = experiment["results"]
    scenarios = list(experiment["scenarios"])
    systems = experiment["systems"]
    print_console_summary(experiment)

    columns, rows = per_vowel_rows(results, systems, scenarios)
    print_table("Akurasi per Vowel", ["Vowel", *[f"{s} {n}" for s, n in columns]], rows)
    for selected, title in [
        (["Closed Recording", "Closed Speakers"], "Akurasi per Speaker: template dan closed"),
        (["Open Speakers"], "Akurasi per Speaker: open"),
    ]:
        selected = [scenario for scenario in selected if scenario in scenarios]
        if not selected:
            continue
        columns, rows = per_speaker_rows(results, selected, systems)
        print_table(title, ["Speaker/group", *[f"{s} {n}" for s, n in columns], "Rata-rata"], rows)
