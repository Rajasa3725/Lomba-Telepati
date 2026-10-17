"""Bangun main.ipynb hasil perbaikan dari main.orig.ipynb (hanya mengubah sel yang perlu)."""
import json
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

WORK = Path(".")
ORIG = WORK / "main.ipynb"  # Gunakan "main.orig.ipynb" jika kamu memiliki file aslinya
OUT = WORK / "main.ipynb"
REPO_YAML = WORK / "yolo26s-p2.yaml"

nb = nbformat.read(ORIG, as_version=4)
cells = nb.cells


def src(i):
    return cells[i].source


# ---------------------------------------------------------------- YAML (ASCII bersih)
yaml_lines = REPO_YAML.read_text(encoding="utf-8").splitlines()
yaml_lines = [ln for ln in yaml_lines if "AGPL-3.0" not in ln]  # buang header ber-mojibake
yaml_text = "\n".join(yaml_lines).strip() + "\n"
yaml_text = (
    "# YOLO26s dengan head deteksi P2/4 - P5/32 (salinan konfigurasi Ultralytics yolo26-p2.yaml, nc=11)\n"
    "# Diinisialisasi dari arsitektur kosong: TIDAK ada bobot pra-latih.\n" + yaml_text
)
assert yaml_text.isascii(), "YAML masih mengandung karakter non-ASCII"
assert "'''" not in yaml_text
assert "nc: 11" in yaml_text

# ---------------------------------------------------------------- sel-sel baru / yang diubah
CELL_1 = '%pip install -q gdown matplotlib pandas numpy pillow "ultralytics==8.4.163"'

CELL_5 = '''import json
import os
import random
import shutil
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def set_seed(seed: int = 42) -> None:
    """Kunci seluruh sumber keacakan: Python, NumPy, PyTorch CPU/CUDA, cuDNN, dan cuBLAS."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


set_seed(42)
'''

# Config: ambil sel asli lalu ubah bagian yang perlu (lebih aman daripada menulis ulang)
cfg_src = src(7)
cfg_src = cfg_src.replace(
    "@dataclass\nclass Config:",
    '''def _pilih_device():
    """GPU tunggal (indeks 0) bila ada, selain itu CPU.

    Multi-GPU (DDP) sengaja TIDAK dipakai: Ultralytics menjalankan DDP di proses baru sehingga patch
    BiFPN/EIoU di notebook tidak ikut terbawa, dan hasil menjadi bergantung pada jumlah GPU.
    """
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class Config:''',
)
old_train_block = cfg_src[cfg_src.index("    # --- Hyperparameter training YOLO26s (From Scratch) ---"): cfg_src.index("    train_project: Path")]
new_train_block = '''    # --- Hyperparameter training YOLO26s-P2 (From Scratch) ---
    model_yaml: str = "yolo26s-p2.yaml"
    train_epochs: int = 100  # lihat estimasi waktu yang dicetak setelah epoch ke-3 (batas sesi Kaggle 12 jam)
    train_batch: int = 16  # nbs=64 -> akumulasi gradien x4 => batch efektif 64 (sama dengan baseline)
    train_imgsz: int = 640
    train_workers: int = 4  # dikunci eksplisit agar augmentasi tidak bergantung pada jumlah CPU
    train_save_period: int = 10
    train_pretrained: bool = False
    train_amp: bool = True
    train_optimizer: str = "AdamW"
    train_lr0: float = 0.001  # default Ultralytics (0.01) terlalu besar untuk AdamW
    train_weight_decay: float = 0.01
    train_warmup_bias_lr: float = 0.0  # default 0.1 terlalu besar untuk Adam/AdamW
    train_hsv_h: float = 0.01
    train_hsv_s: float = 0.3
    train_hsv_v: float = 0.3
    train_translate: float = 0.2
    train_shear: float = 0.0
    train_perspective: float = 0.0
    train_scale: float = 0.4
    train_degrees: float = 0.0
    train_fliplr: float = 0.3
    train_flipud: float = 0.3
    train_mosaic: float = 0.4
    train_close_mosaic: int = 10
    train_erasing: float = 0.1
    train_auto_augment: Optional[str] = None
    train_cos_lr: bool = True
    train_patience: int = 0  # 0 = tanpa early stopping (seluruh epoch dijalankan)
    train_device: Any = field(default_factory=_pilih_device)
'''
cfg_src = cfg_src.replace(old_train_block, new_train_block)
cfg_src = cfg_src.replace(
    "    @property\n    def class_to_id",
    "    # Diisi otomatis setelah training selesai (jalur pasti best.pt dari trainer)\n    best_override: Optional[str] = None\n\n    @property\n    def class_to_id",
)
cfg_src = cfg_src.replace(
    "    def best_weights_path(self) -> str:\n        candidates = [",
    "    def best_weights_path(self) -> str:\n        if self.best_override and Path(self.best_override).exists():\n            return self.best_override\n        candidates = [",
)
assert "best_override" in cfg_src and "_pilih_device" in cfg_src and "train_lr0" in cfg_src
assert "train_device: Any" in cfg_src

CELL_13 = '''def create_yolo_dirs(cfg: Config) -> None:
    # Bersihkan keluaran lama agar eksekusi ulang identik dengan eksekusi pertama
    # (tanpa sisa label/cache). Hanya images/ dan labels/ yang dihapus, folder hasil training tidak disentuh.
    for sub in ("images", "labels"):
        shutil.rmtree(cfg.yolo_out / sub, ignore_errors=True)

    for split in ["train", "val", "test"]:
        (cfg.yolo_out / "images" / split).mkdir(parents=True, exist_ok=True)
        (cfg.yolo_out / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("Folder YOLO berhasil dibuat.")


create_yolo_dirs(CFG)
'''

LEAK_MD = """## A.11 Diagnostik Duplikasi Antar-Split (Data Leakage)
Dataset berasal dari ekspor Roboflow: satu gambar sumber dapat muncul sebagai beberapa salinan hasil augmentasi
(nama berpola `<sumber>_jpg.rf.<hash>.jpg`). Sel ini hanya **melaporkan** apakah gambar sumber yang sama muncul di
lebih dari satu split (berdasarkan nama sumber dan hash isi file). Split resmi panitia tetap dipakai apa adanya,
tidak ada data yang dibuang atau dipindah."""

LEAK_CODE = '''import hashlib
import re


def _nama_sumber(file_name: str) -> str:
    """Buang akhiran Roboflow `_jpg.rf.<hash>` agar salinan augmentasi dari satu sumber dikenali."""
    return re.split(r"_jpg\\.rf\\.|\\.rf\\.", Path(file_name).name)[0]


def cek_kebocoran_antar_split(cfg: Config) -> None:
    sumber: Dict[str, set] = {}
    hash_isi: Dict[str, set] = {}

    for split in cfg.coco_splits:
        json_path = cfg.dataset_root / split / cfg.coco_annotation_file
        coco = json.loads(json_path.read_text(encoding="utf-8"))
        nama = [im["file_name"] for im in coco["images"]]
        sumber[split] = {_nama_sumber(n) for n in nama}

        hashes = set()
        for n in nama:
            p = cfg.dataset_root / split / n
            if p.exists():
                hashes.add(hashlib.md5(p.read_bytes()).hexdigest())
        hash_isi[split] = hashes
        print(f"{split:<6}: {len(nama):>6} gambar | {len(sumber[split]):>6} sumber unik")

    print("\\nIrisan antar-split:")
    pasangan = [("train", "valid"), ("train", "test"), ("valid", "test")]
    for a, b in pasangan:
        irisan_nama = sumber[a] & sumber[b]
        irisan_hash = hash_isi[a] & hash_isi[b]
        print(f"  {a:>5} & {b:<5}: sumber sama = {len(irisan_nama):>5} | isi file identik = {len(irisan_hash):>5}")


cek_kebocoran_antar_split(CFG)
'''

YAML_MD = """## B.0 Arsitektur: YOLO26s dengan Head P2 (dari Konfigurasi YAML)
Model dibangun dari **arsitektur kosong** lewat file konfigurasi YAML (tanpa bobot pra-latih). Konfigurasi di bawah
menambahkan jalur fitur stride-4 (P2, 160x160 pada input 640) ke neck dan head deteksi, agar lesi mikro seperti
`brown_spot` (median kurang dari 0,3% luas citra) tidak hilang saat downsampling. File ditulis dari dalam notebook
sehingga arsitektur terlihat utuh di sini dan tidak bergantung pada file di luar notebook."""

YAML_CODE = "YOLO26S_P2_YAML = r'''" + yaml_text + "'''\n\nPath(CFG.model_yaml).write_text(YOLO26S_P2_YAML, encoding=\"utf-8\")\nprint(\"Arsitektur ditulis ke:\", Path(CFG.model_yaml).resolve())\n"

PATCH_MD = """## B.1 Modul Kustom: BiFPN Fast Normalized Fusion & Loss EIoU
1. **`BiFPN_Concat`**: fusi fitur berbobot terpelajari $w_i \\ge 0$ pada setiap `Concat` di head:
   $$O = \\text{Concat}\\left(\\frac{w_1}{\\epsilon + w_1 + w_2} \\cdot x_1, \\frac{w_2}{\\epsilon + w_1 + w_2} \\cdot x_2\\right)$$
2. **EIoU** menggantikan CIoU di `BboxLoss`; penalti lebar dan tinggi dipisah agar lesi tipis memanjang
   (`bacterial_leaf_blight`) tidak terdistorsi gradiennya:
   $$\\mathcal{L}_{EIoU} = 1 - IoU + \\frac{\\rho^2(b, b^{gt})}{c^2} + \\frac{\\rho^2(w, w^{gt})}{C_w^2} + \\frac{\\rho^2(h, h^{gt})}{C_h^2}$$
3. **Lewati pengecekan AMP bawaan**: sebelum training, Ultralytics menjalankan `check_amp` yang mengunduh `yolo26n.pt`
   hanya untuk membandingkan keluaran FP16 dan FP32. Bobot itu **tidak pernah dipakai untuk melatih**, tetapi unduhannya
   tampak di log dan tidak sesuai semangat aturan "tanpa bobot pra-latih". Pengecekan ini dilewati (AMP tetap aktif di GPU)
   sehingga tidak ada bobot bawaan yang diunduh.

> Patch dipasang di dalam proses notebook. Karena itu training memakai **satu GPU** (lihat `_pilih_device`)."""

PATCH_CODE = '''import torch
import torch.nn as nn
import ultralytics
import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks
import ultralytics.utils.loss as loss_module
import ultralytics.engine.trainer as trainer_module

EXPECTED_ULTRALYTICS = "8.4.163"
print("ultralytics", ultralytics.__version__, "| torch", torch.__version__)
if ultralytics.__version__ != EXPECTED_ULTRALYTICS:
    print(
        f"PERINGATAN: kode ini diuji pada ultralytics=={EXPECTED_ULTRALYTICS}; versi lain dapat "
        "mengubah nama internal yang di-patch dan membuat hasil audit berbeda."
    )


class BiFPN_Concat(modules.Concat):
    """BiFPN Fast Normalized Feature Fusion untuk dua masukan."""

    def __init__(self, dimension=1):
        super().__init__(dimension=dimension)
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 1e-4

    def forward(self, x):
        if len(x) == 2:
            w = torch.relu(self.w)
            norm_w = w / (torch.sum(w) + self.epsilon)
            return torch.cat([norm_w[0] * x[0], norm_w[1] * x[1]], self.d)
        return super().forward(x)


def bbox_iou_eiou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """Mengembalikan `IoU - penalti EIoU` (loss = 1 - nilai ini), kompatibel dengan ultralytics.utils.metrics.bbox_iou."""
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
    rho_w2 = (w2 - w1).pow(2)
    rho_h2 = (h2 - h1).pow(2)

    return iou - (rho2 / c2 + rho_w2 / (cw.pow(2) + eps) + rho_h2 / (ch.pow(2) + eps))


def _check_amp_tanpa_unduhan(model) -> bool:
    """Pengganti `check_amp`: tetap menghormati perangkat (AMP mati di CPU/MPS) tetapi tanpa mengunduh bobot."""
    return next(model.parameters()).device.type not in {"cpu", "mps"}


def terapkan_patch_kustom() -> None:
    """Pasang modul kustom ke mesin Ultralytics (idempoten; aman dipanggil berulang)."""
    assert hasattr(loss_module, "bbox_iou") and hasattr(tasks, "Concat") and hasattr(trainer_module, "check_amp"), (
        "Struktur internal Ultralytics berbeda dari 8.4.163; patch tidak dapat dipasang."
    )
    tasks.Concat = BiFPN_Concat  # parse_model membaca nama global ini saat membangun head
    loss_module.bbox_iou = bbox_iou_eiou  # dipanggil oleh BboxLoss.forward
    trainer_module.check_amp = _check_amp_tanpa_unduhan
    assert tasks.Concat is BiFPN_Concat and loss_module.bbox_iou is bbox_iou_eiou


terapkan_patch_kustom()
print("BiFPN_Concat, EIoU, dan pelewatan check_amp terpasang.")
'''

TRAINER_MD = """## B.2 Trainer Kustom: Inisialisasi Kaiming Normal & Pagar Pengaman From-Scratch
Ultralytics membangun modelnya sendiri di dalam trainer dari nama file YAML, sehingga model yang dibuat dan
diinisialisasi di luar trainer **tidak pernah dipakai**. Karena itu inisialisasi dipasang di dalam `ScratchTrainer.get_model()`,
tepat setelah model dibangun dan sebelum optimizer dibuat.

- Kaiming Normal (`fan_out`, `relu`) diterapkan ke semua `Conv2d` **kecuali head `Detect`**. Head dikecualikan karena
  `Detect.bias_init()` menetapkan prior bias kelas dan box; menolkannya akan memperlambat konvergensi awal.
- `get_model()` menolak bobot apa pun yang dimuat (kecuali melanjutkan checkpoint buatan sendiri saat `resume`),
  sehingga pelatihan tidak mungkin tanpa sengaja memakai transfer learning.
- `get_model()` memastikan ada tepat **6 modul `BiFPN_Concat`** pada model; bila patch tidak aktif, training berhenti
  seketika dengan pesan jelas, alih-alih berjalan diam-diam tanpa BiFPN."""

TRAINER_CODE = '''from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import Detect

N_BIFPN_DIHARAPKAN = 6  # jumlah Concat pada head yolo26-p2


def init_kaiming_weights(model) -> int:
    """Kaiming Normal pada seluruh Conv2d kecuali head Detect. Mengembalikan jumlah lapisan yang diinisialisasi."""
    param_head = {id(p) for m in model.modules() if isinstance(m, Detect) for p in m.parameters()}
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and id(m.weight) not in param_head:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            n += 1
    return n


class ScratchTrainer(DetectionTrainer):
    """DetectionTrainer yang memastikan model dilatih dari nol dengan modul kustom aktif."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        if weights is not None and not getattr(self, "resume", False):
            raise RuntimeError("Bobot pra-latih terdeteksi: dilarang oleh aturan lomba (wajib from scratch).")
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        if weights is None:
            n_init = init_kaiming_weights(model)
            print(f"[INIT] Kaiming Normal diterapkan pada {n_init} lapisan Conv2d (head Detect dikecualikan).")
        n_bifpn = sum(isinstance(m, BiFPN_Concat) for m in model.modules())
        if n_bifpn != N_BIFPN_DIHARAPKAN:
            raise RuntimeError(
                f"BiFPN_Concat aktif pada {n_bifpn} modul (seharusnya {N_BIFPN_DIHARAPKAN}). "
                "Pastikan sel B.1 sudah dijalankan dan training memakai satu GPU."
            )
        if loss_module.bbox_iou is not bbox_iou_eiou:
            raise RuntimeError("Loss EIoU tidak aktif (bbox_iou belum di-patch).")
        print(f"[CEK] BiFPN_Concat = {n_bifpn} modul | loss = EIoU | bobot pra-latih = tidak ada")
        return model


def _estimasi_waktu(trainer) -> None:
    """Setelah epoch ke-3, cetak estimasi total waktu agar batas sesi Kaggle (12 jam) bisa diantisipasi dini."""
    if trainer.epoch - trainer.start_epoch + 1 == 3:
        rata2 = (time.time() - trainer.train_time_start) / 3
        total_jam = rata2 * trainer.epochs / 3600
        print(f"[WAKTU] ~{rata2 / 60:.1f} menit/epoch -> estimasi {total_jam:.1f} jam untuk {trainer.epochs} epoch.")
        if total_jam > 11:
            print("[WAKTU] PERINGATAN: melebihi ~11 jam. Hentikan sekarang dan turunkan `train_epochs` di Config.")
'''

TRAIN_MD = """## B.3 Pelatihan Model (From Scratch, GPU Tunggal)
Optimizer **AdamW** dengan `lr0=0.001`, `weight_decay=0.01`, `warmup_bias_lr=0.0`, dijadwalkan cosine. Nilai ini
ditetapkan eksplisit karena bawaan Ultralytics (`lr0=0.01`, `warmup_bias_lr=0.1`) ditujukan untuk SGD dan terlalu besar
untuk AdamW. Semua seed dikunci (`seed=42`, `deterministic=True`) dan `workers` ditetapkan tetap.

Set `RUN_TRAINING = False` hanya bila melanjutkan evaluasi dari `best.pt` buatan sendiri yang sudah ada;
untuk Audit Run, biarkan `True` agar seluruh alur berjalan dari data mentah sampai evaluasi."""

TRAIN_CODE = '''from ultralytics import YOLO


def train_model(cfg: Config):
    dev = cfg.train_device
    if (isinstance(dev, (list, tuple)) and len(dev) > 1) or (isinstance(dev, str) and "," in dev):
        raise RuntimeError(
            "Multi-GPU (DDP) tidak didukung: patch BiFPN/EIoU hanya berlaku di proses notebook. "
            "Gunakan satu GPU (train_device=0)."
        )

    terapkan_patch_kustom()  # pastikan patch aktif di proses ini

    # 1. Arsitektur kosong dari YAML (tanpa bobot pra-latih)
    model = YOLO(cfg.model_yaml)
    model.add_callback("on_fit_epoch_end", _estimasi_waktu)

    # 2. Pelatihan dengan trainer kustom (Kaiming init + pagar pengaman) -- AdamW + cosine
    results = model.train(
        trainer=ScratchTrainer,
        data=str(cfg.yolo_out / "data.yaml"),
        epochs=cfg.train_epochs,
        batch=cfg.train_batch,
        imgsz=cfg.train_imgsz,
        workers=cfg.train_workers,
        save_period=cfg.train_save_period,
        pretrained=cfg.train_pretrained,
        seed=cfg.seed,
        deterministic=True,
        amp=cfg.train_amp,
        optimizer=cfg.train_optimizer,
        lr0=cfg.train_lr0,
        weight_decay=cfg.train_weight_decay,
        warmup_bias_lr=cfg.train_warmup_bias_lr,
        cos_lr=cfg.train_cos_lr,
        patience=cfg.train_patience,
        # HSV Augmentation
        hsv_h=cfg.train_hsv_h,
        hsv_s=cfg.train_hsv_s,
        hsv_v=cfg.train_hsv_v,
        # Geometric Augmentation
        translate=cfg.train_translate,
        shear=cfg.train_shear,
        perspective=cfg.train_perspective,
        scale=cfg.train_scale,
        degrees=cfg.train_degrees,
        # Flip & Mosaic
        fliplr=cfg.train_fliplr,
        flipud=cfg.train_flipud,
        mosaic=cfg.train_mosaic,
        close_mosaic=cfg.train_close_mosaic,
        erasing=cfg.train_erasing,
        auto_augment=cfg.train_auto_augment,
        project=str(cfg.train_project),
        name=cfg.train_name,
        device=cfg.train_device,
        exist_ok=True,
    )

    # 3. Simpan jalur pasti best.pt dari trainer untuk tahap evaluasi
    best = Path(model.trainer.best)
    cfg.best_override = str(best if best.exists() else model.trainer.last)
    print("Bobot terbaik:", cfg.best_override)
    return model, results


RUN_TRAINING = True
if RUN_TRAINING:
    model_trained, results_trained = train_model(CFG)
'''

EVAL_VAL_CODE = '''def evaluate_model(model: YOLO, cfg: Config, split: str):
    return model.val(
        data=str(cfg.yolo_out / "data.yaml"),
        split=split,
        imgsz=cfg.train_imgsz,
        batch=16,
        workers=cfg.train_workers,
        device=cfg.train_device,
    )


def print_per_class(metrics, names: List[str]) -> None:
    """Tabel metrik per kelas: kelas lemah (mis. brown_spot) langsung terlihat."""
    print(f"{'kelas':<26}{'P':>7}{'R':>7}{'mAP50':>8}{'mAP50-95':>10}")
    for i, c in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        print(f"{names[int(c)]:<26}{p:>7.3f}{r:>7.3f}{ap50:>8.3f}{ap:>10.3f}")


metrics = evaluate_model(best_model, CFG, split="val")

print("mAP50 :", metrics.box.map50)
print("mAP50-95 :", metrics.box.map)
print()
print_per_class(metrics, CFG.class_names)
'''

EVAL_TEST_ADD = '''
print()
print_per_class(final_metrics, CFG.class_names)
'''

AUDIT_MD = """## C.4 Pemeriksaan Kepatuhan: Tidak Ada Bobot Bawaan Ultralytics
Memastikan tidak ada berkas bobot pra-latih (mis. `yolo26n.pt`) yang terunduh atau tertinggal di direktori kerja.
Berkas `best.pt`, `last.pt`, dan `epochN.pt` hasil pelatihan sendiri adalah wajar."""

AUDIT_CODE = '''def audit_bobot_pretrained(root: Path = Path.cwd()) -> None:
    from ultralytics.utils import WEIGHTS_DIR

    nama_bawaan = ("yolo26", "yolo11", "yolov8", "yolov5", "yolov10", "yolov9", "rtdetr", "sam")
    kandidat = list(root.rglob("*.pt")) + list(Path(WEIGHTS_DIR).glob("*.pt"))
    mencurigakan = sorted({str(p) for p in kandidat if p.name.lower().startswith(nama_bawaan)})
    if mencurigakan:
        print("PERINGATAN: berkas bobot bawaan ditemukan, hapus sebelum submit:")
        for p in mencurigakan:
            print("  -", p)
    else:
        print("OK: tidak ada berkas bobot bawaan Ultralytics yang terunduh atau tertinggal.")


audit_bobot_pretrained()
'''

# ---------------------------------------------------------------- rakit sel
new_cells = []
for i, c in enumerate(cells):
    if i == 1:
        c.source = CELL_1
    elif i == 5:
        c.source = CELL_5
    elif i == 7:
        c.source = cfg_src
    elif i == 13:
        c.source = CELL_13
    elif i == 36:
        c.source = EVAL_VAL_CODE
    elif i == 38:
        c.source = src(38) + EVAL_TEST_ADD
    elif i == 34:
        c.source = src(34)

    if i in range(25, 32):
        continue  # blok B lama diganti seluruhnya

    new_cells.append(c)

    if i == 23:  # setelah distribusi kelas -> diagnostik leakage
        new_cells.append(new_markdown_cell(LEAK_MD))
        new_cells.append(new_code_cell(LEAK_CODE))
    if i == 24:  # setelah pemisah '---' -> blok B baru
        new_cells.append(new_markdown_cell("### Bagian B — Arsitektur & Pelatihan (From Scratch)"))
        new_cells.append(new_markdown_cell(YAML_MD))
        new_cells.append(new_code_cell(YAML_CODE))
        new_cells.append(new_markdown_cell(PATCH_MD))
        new_cells.append(new_code_cell(PATCH_CODE))
        new_cells.append(new_markdown_cell(TRAINER_MD))
        new_cells.append(new_code_cell(TRAINER_CODE))
        new_cells.append(new_markdown_cell(TRAIN_MD))
        new_cells.append(new_code_cell(TRAIN_CODE))

new_cells.append(new_markdown_cell(AUDIT_MD))
new_cells.append(new_code_cell(AUDIT_CODE))
nb.cells = new_cells

# ---------------------------------------------------------------- bersihkan output lama + metadata run lama
for c in nb.cells:
    if c.cell_type == "code":
        c.outputs = []
        c.execution_count = None
nb.metadata.pop("papermill", None)

# pastikan id sel unik
seen = set()
for k, c in enumerate(nb.cells):
    if not c.get("id") or c["id"] in seen:
        c["id"] = f"cell_{k:03d}"
    seen.add(c["id"])

nbformat.validate(nb)
nbformat.write(nb, OUT)
print("OK ->", OUT, "| jumlah sel:", len(nb.cells))
