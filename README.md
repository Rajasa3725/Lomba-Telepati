# Telepati Agridata - AI AgriData Intelligence Race (Telepati 8.0)

Repositori ini dikembangkan untuk perlombaan **AI AgriData Intelligence Race (Telepati 8.0)** dengan fokus pada deteksi multi-kelas penyakit daun padi (11 kelas kanonikal) menggunakan arsitektur Computer Vision / Object Detection yang dilatih murni dari awal (*from scratch*).

---

## 📋 Regulasi Kepatuhan & Karakteristik Model

1. **No Pretrained / From Scratch:** Model diinisialisasi dan dilatih 100% dari awal via konfigurasi YAML, mematuhi larangan transfer learning.
2. **11 Kelas Kanonikal Resmi:** Melakukan standarisasi dan pemetaan 18 label mentah COCO menjadi 11 kelas resmi.
3. **Reproduksibilitas Deterministik:** Kunci 4 lapis seed (Python, NumPy, PyTorch CPU, CUDA, dan CuDNN deterministic).
4. **Single Pipeline:** Seluruh alur end-to-end terangkum di dalam satu file `main.ipynb`.

---

## 🚀 Cara Menjalankan Proyek

### 1. Prasyarat

Pastikan Python 3.12 atau lebih baru dan `uv` sudah terinstal di sistem Anda.

Instal `uv` sesuai sistem operasi:

* **macOS / Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

* **Windows PowerShell:**
  ```powershell
  powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

### 2. Instalasi Dependensi

1. Clone repositori dan masuk ke direktori kerja:
   ```bash
   git clone https://github.com/Rajasa3725/Lomba-Telepati.git
   cd Lomba-Telepati
   ```

2. Sinkronkan environment dan dependensi dengan `uv`:
   ```bash
   uv sync
   ```

3. Jalankan Jupyter Lab:
   ```bash
   uv run --with jupyter jupyter lab
   ```

### 3. Eksekusi Notebook (`main.ipynb`)

* Buka `main.ipynb` di Jupyter Lab atau VS Code (pilih kernel Python dari `.venv` yang dibuat oleh `uv`).
* Kode konfigurasi telah dilengkapi deteksi path hibrida (otomatis menyesuaikan lingkungan Kaggle `/kaggle/working` atau lokal VS Code).
* Jalankan seluruh sel secara berurutan (*Run All*).

---

## 📊 Spesifikasi Model & Hyperparameter Pelatihan

* **Arsitektur Model:** `yolo26s` (From Scratch)
* **Resolusi Input:** 640×640
* **Target Output:** 11 Kelas Kanonikal
* **Optimasi:** AdamW / MuSGD, Epochs: 200, Seed: 42
* **Tautan Unduh Bobot Resmi:** Bobot tersimpan via GitHub Releases / Git LFS.
