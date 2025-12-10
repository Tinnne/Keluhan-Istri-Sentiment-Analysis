# Sentimen Analysis on Keluhan Istri Data

- **Dataset:** Data Keluhan Istri (Data/RawData.csv)
- **Tasks:** Indonesian Text Cleaning, Sentiment Classification (multi-class)
- **Metrics:** Macro F1 Score
- **Goal:** Membangun model klasifikasi sentimen untuk teks keluhan istri berbahasa Indonesia (informal & campur-campur), dengan fokus pada preprocessing bahasa Indonesia dan evaluasi menggunakan F1 (macro) agar kelas minoritas tetap terjaga performanya.

## 📌 Overview

- Teks keluhan bersifat panjang, emosional, dan sangat informal (slang, campuran bahasa, emoji).
- **Tujuan:** membangun model yang bisa mengklasifikasikan keluhan ke beberapa label sentimen sehingga bisa dianalisis lebih lanjut (misalnya untuk insight psikologis/sosial).
- Workflow utama:
  1. EDA awal.
  2. Pembersihan & normalisasi teks bahasa Indonesia.
  3. TF-IDF feature extraction.
  4. Modeling dengan XGBoost dan ANN (PyTorch + Optuna).
  5. Evaluasi dengan macro F1.

## 📊 Exploratory Data Analysis

Notebook: 01EDA.ipynb & 03CleanedEDA.ipynb

- ±3k baris teks keluhan dengan label numerik.
- Tidak ada missing value di kolom utama, tapi ada duplikasi yang di-drop pada tahap cleaning.
- EDA mencakup:
  - Distribusi kelas label.
  - Panjang teks (kata/karakter).
  - Kata & n-gram yang sering muncul (mis. via WordCloud).

## 🧹 Text Cleaning (Bahasa Indonesia)

**Notebook:** 02DataCleaning.ipynb

**Langkah utama:**
- Lowercase.
- Menghapus emoji, URL, angka tak relevan, dan karakter non-alfabet.
- Normalisasi spasi dan line break.
- Stopwords bahasa Indonesia + beberapa kata domain-spesifik.
- Normalisasi slang (misal variasi “bund/bunda/bun”) dengan kamus.

**Output:** Clean Data (Data/CleanData.csv) berisi kolom text (bersih) dan label.

## 🤖 Modeling

### Feature Representation:
- TF-IDF (unigram + n-gram) menggunakan TfidfVectorizer (scikit-learn).

### Models:
**Notebook:** 04XGBModel.ipynb & 05ANNModel.ipynb
- Baseline: TF-IDF + XGBClassifier
- Advance: TF-IDF + ANN

### Hyperparameter Tuning
- GridSearchCV untuk hyperparameter tuning

## 📈 Results & Interpretation

- Evaluasi utama: macro F1 untuk menjaga performa di kelas minoritas.
- Analisis di notebook mencakup:
  - Classification report & confusion matrix.
  - Contoh teks yang salah klasifikasi.
- Insight yang bisa di-highlight:
  - Kelas yang paling sering tertukar.
  - N-gram yang paling kuat terkait kelas tertentu.
