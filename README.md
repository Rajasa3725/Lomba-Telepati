<<<<<<< HEAD
# Telepati Agridata

Proyek ini dikembangkan untuk perlombaan **Telepati Agridata**. Repositori ini berisi notebook eksperimen, pemrosesan dataset, dan persiapan model *computer vision* serta *object detection*.

## Cara Menjalankan Proyek

### Prasyarat

Pastikan Python 3.12 atau lebih baru dan `uv` sudah terinstal.

Instal `uv` sesuai sistem operasi Anda:

**macOS/Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows PowerShell**

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Instalasi

1. Clone repositori, lalu masuk ke direktorinya:

  ```bash
  git clone https://github.com/Isannali/lomba.git
  cd lomba
  ```

2. Sinkronkan environment dan dependensi:

  ```bash
  uv sync
  ```

3. Jalankan Jupyter Lab:

  ```bash
  uv run --with jupyter jupyter lab
  ```

### Jika Menggunakan VS Code

Buka repositori ini di VS Code, lalu pilih interpreter atau kernel Python dari environment `.venv` yang dibuat oleh `uv`.

### Pengaturan Dataset
1. Unduh dataset lomba terlebih dahulu ke komputer Anda.
2. Ekstrak file dataset tersebut ke direktori pilihan Anda.
3. Buka notebook eksperimen (.ipynb) di Jupyter Lab atau VS Code.
4. Cari variabel DATASET_ROOT dan YOLO_OUT di cell awal notebook, lalu ubah nilainya sesuai lokasi/path folder dataset sesuai dengan Path pada folder anda.
contoh:
```python
dataset_root = Path(r"path/to/your/extracted_dataset")
yolo_out = Path(r"path/to/your/destination")
```
5. Jalankan seluruh cell secara berurutan dari atas ke bawah (`Run All`).



=======
## Setup & Cara Menjalankan Project

Proyek ini dikembangkan menggunakan **Python 3.11+** dan **PyTorch (CUDA 12.4)**. Follow langkah-langkah di bawah untuk menyiapkan environment dan menjalankan skrip.

### 1. Prerequisites
* Python versi 3.11 atau yang lebih baru.
* GPU NVIDIA dengan driver CUDA yang sudah terinstall (opsional untuk akselerasi training/inference).

---

### 2. Instalasi Environment (Local Machine / Remote Server)

Buka terminal / PowerShell di folder proyek ini:

1. **Buat & Aktifkan Virtual Environment:**
   * **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\activate
     ```
   * **Linux / macOS /:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

2. **Update `pip`:**
   ```bash
   python -m pip install --upgrade pip

3. pip install -r requirements.txt

--extra-index-url https://download.pytorch.org/whl/cu124

torch>=2.2.0
torchvision>=0.17.0
ultralytics>=8.1.0
albumentations<2.0.0
stringzilla
pandas
numpy
ipykernel
>>>>>>> 4b541eb059fde940d32930716c59b564b2148ee9
