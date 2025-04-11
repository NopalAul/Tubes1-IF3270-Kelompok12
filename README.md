# Tubes1-IF3270-Kelompok12: Implementasi Feedforward Neural Network (FFNN)

Repository ini berisi implementasi Feedforward Neural Network (FFNN) *from scratch* untuk tugas mata kuliah Pembelajaran Mesin (IF3270).

## Project Overview

Pada proyek ini, kami mengimplementasikan Feedforward Neural Network (FFNN) dari awal menggunakan NumPy. Implementasi ini mencakup:

- Berbagai fungsi aktivasi (ReLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU)
- Fungsi loss (MSE, Binary dan Categorical Cross Entropy)
- Metode inisialisasi bobot (Zero, Uniform, Normal, Xavier, He)
- Teknik regularisasi (L1, L2)
- Metode normalisasi (RMS Norm)
- Training dengan gradient descent dan backpropagation

## Requirements

Untuk menjalankan proyek ini, Anda memerlukan:

- Python 3.8 atau lebih baru
- NumPy
- Matplotlib
- scikit-learn
- tqdm
- networkx

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/NopalAul/Tubes1-IF3270-Kelompok12.git
    cd Tubes1-IF3270-Kelompok12
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Untuk menggunakan implementasi FFNN ini, anda bisa mengujinya dengan menjalankan Notebook Jupyter yang telah disediakan di dalam folder src.

- Untuk pengujian model FFNN dapat menjalankan `src/pengujian.ipynb`

- Untuk pengujian implementasi Normalisasi dapat menjalankan `src/pengujianNormalisasi.ipynb`.

- Untuk pengujian implementasi Regularisasi dapat menjalankan `src/pengujianRegularisasi.ipynb`.

- sebagai pembanding dapat menjalankan `src/pengujianSklearn.ipynb`.

## Pembagian Tugas

|                  Kegiatan                  |                                       Nama (NIM)                                      |
|:------------------------------------------:|:-------------------------------------------------------------------------------------:|
|        Implementasi fungsi aktivasi        |              Muhammad Naufal Aulia (13522074) Dhidit Abdi Aziz (13522040)             |
|          Implementasi fungsi loss          |                            Muhammad Naufal Aulia (13522074)                           |
|      Implementasi layer neural network     |                                Tazkia Nizami (13522032)                               |
|      Implementasi metode inisialisasi      |                            Muhammad Naufal Aulia (13522074)                           |
|   Implementasi dan Pengujian Regularisasi  |                              Dhidit Abdi Aziz (13522040)                              |
| Implementasi dan Pengujian Normalisasi RMS |                              Dhidit Abdi Aziz (13522040)                              |
|         Visualisasi informasi model        |               Tazkia Nizami (13522032) Muhammad Naufal Aulia (13522074)               |
|            Integrasi model FFNN            |                            Muhammad Naufal Aulia (13522074)                           |
|       Implementasi notebook pengujian      |               Tazkia Nizami (13522032) Muhammad Naufal Aulia (13522074)               |
|               Pengujian model              | Tazkia Nizami (13522032) Dhidit Abdi Aziz (13522040) Muhammad Naufal Aulia (13522074) |
|             Pengerjaan laporan             | Tazkia Nizami (13522032) Dhidit Abdi Aziz (13522040) Muhammad Naufal Aulia (13522074) |
