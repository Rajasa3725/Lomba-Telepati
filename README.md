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