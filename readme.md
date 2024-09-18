# **Adaptive Bitrate Prediction Model**

## Deskripsi

Project ini bertujuan untuk mengembangkan model prediksi adaptive bitrate pada sistem streaming video dengan menggunakan pendekatan machine learning. Model yang berhasil diimplementasikan adalah Long Short-Term Memory (LSTM) dan Gated Recurrent Unit (GRU), yang keduanya dikenal efektif dalam menangani data sekuensial seperti pola jaringan pada streaming.

## Dataset

Dataset yang digunakan adalah file dengan format CSV, yang berisi data sekuensial terkait kondisi jaringan dan video.

# Struktur Dataset

Dataset yang digunakan memiliki struktur sebagai berikut:

## Fitur

- resolutions
- estimatedBandwidths
- streamBandwidths
- bufferingStates

## Target

- newBitrates

## Setup

### Persyaratan

- Python 3.x
- Jupyter Notebook
- Library Python:
  - TensorFlow/Keras (untuk implementasi model LSTM dan GRU)
  - Pandas (untuk manipulasi data)
  - Numpy (untuk perhitungan numerik)
  - Scikit-learn (untuk preprocessing data)
  - Matplotlib/Seaborn (untuk visualisasi hasil)

// Start Generation Here

## Instalasi

Untuk menginstal semua dependensi yang diperlukan, Anda dapat menggunakan pip. Jalankan perintah berikut di terminal Anda:

### Clone repo ini ke lokal:

```bash
git clone https://github.com/username/adaptive-bitrate-model.git
cd adaptive-bitrate-model
```

### Buat virtual environment:

```bash
python -m venv env
source env/bin/activate # untuk macOS/Linux
# atau
.\env\Scripts\activate # untuk Windows
```

### Install dependencies:

```bash
pip install -r requirements.txt
```
