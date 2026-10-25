# Vowel ASR: DTW vs Gaussian

Pengenalan 5 vowel Bahasa Indonesia (/a/ /i/ /u/ /e/ /o/) dengan fitur MFCC 39 dimensi
menggunakan dua sistem:

1. Template-based ASR dengan Dynamic Time Warping (averaged template)
2. ASR berbasis Model Gaussian (mean + covariance, negative log-likelihood)

Praktikum IF4071 Pemrosesan Ucapan, minggu ke-9.

## Struktur

```
data/
  template/      rekaman ke-1 tiap anggota (data latih / template)
  test/closed/   rekaman ke-2 anggota yang sama (closed speakers)
  test/open/     rekaman dari kelompok lain (open speakers)
preprocessing.py load wav + label dari nama file
features.py      ekstraksi MFCC 39 dimensi
dtw.py           DTW, averaged template, recognition
gaussian.py      model Gaussian, negative log-likelihood, recognition
evaluation.py    pipeline eksperimen, metrik, dan confusion matrix
reporting.py     tabel hasil dan ekspor prediksi CSV
main.py          menjalankan eksperimen tanpa notebook
dtw_gaussian.ipynb antarmuka eksplorasi/visualisasi yang memanggil modul .py
```

## Format rekaman

- File WAV; audio stereo dikonversi ke mono dan sample rate diubah ke 16 kHz bila perlu
- Nama file: `<speaker>_<vowel>.wav`, contoh `S01_a.wav`
- Speaker kelompok sendiri `S01`–`S05`, speaker kelompok lain `O01`, `O02`, ...

## Menjalankan

```bash
pip install -r requirements.txt
python main.py
```

`main.py` tidak bergantung pada notebook dan dapat dijalankan dari direktori mana pun.
Secara default, hasil prediksi disimpan ke `outputs/predictions.csv`. Opsi yang tersedia:

```bash
python main.py --root C:/path/ke/proyek
python main.py --output outputs/hasil.csv
python main.py --no-csv
```

Notebook hanya mengorkestrasi pemanggilan fungsi pada modul Python dan menampilkan
tabel/visualisasi. Jalankan dari root repositori, atau isi `PROJECT_ROOT` di sel awal.

## Skenario pengujian

| Skenario | Data uji |
|---|---|
| Closed recordings | `data/template/` (data yang sama dengan template) |
| Closed speakers | `data/test/closed/` |
| Open speakers | `data/test/open/` |

## Anggota dan pembagian tugas

| Nama | NIM | Tugas |
|---|---|---|
| | | Dataset + `preprocessing.py` |
| | | `features.py` |
| | | `dtw.py` |
| | | `gaussian.py` |
| | | `main.py` + perhitungan akurasi |
