"""
====================================================================
config.py — Konfigurasi Global Vision-to-Audio Bridge
====================================================================
Semua konstanta numerik dan definisi warna HSV disentralisasi
di sini agar mudah dikalibrasi tanpa menyentuh logika utama.

Versi perbaikan:
  - Rp20.000 dibuat lebih mudah terdeteksi saat warna hijau pucat/redup
  - Rp2.000 dibuat lebih sulit menang karena abu-abu sering false positive
  - Ditambahkan parameter preprocessing dan template threshold
====================================================================
"""

import numpy as np

# ------------------------------------------------------------------ #
#  DIMENSI FRAME KAMERA                                              #
# ------------------------------------------------------------------ #

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# ------------------------------------------------------------------ #
#  PARAMETER DETEKSI ROI                                             #
# ------------------------------------------------------------------ #

ASPEK_RASIO_MIN = 1.25
ASPEK_RASIO_MAX = 3.8

LUAS_ROI_MIN = 6_000
LUAS_ROI_MAX = 220_000

# ------------------------------------------------------------------ #
#  PARAMETER ROI GUIDE / MANUAL                                      #
# ------------------------------------------------------------------ #

GUIDE_X_RATIO = 0.08
GUIDE_Y_RATIO = 0.26
GUIDE_W_RATIO = 0.84
GUIDE_H_RATIO = 0.48

# Untuk demo, biarkan False agar uang asli tidak gampang ditolak
VALIDASI_KETAT_GUIDE = False

# ------------------------------------------------------------------ #
#  PARAMETER PREPROCESSING                                           #
# ------------------------------------------------------------------ #

# Aktifkan preprocessing agar citra lebih stabil terhadap pencahayaan
GUNAKAN_PREPROCESSING = True

# Gamma > 1 membantu mencerahkan gambar yang redup
GAMMA_VALUE = 1.25

# Saturation boost membantu warna uang yang pucat agar lebih terlihat
SATURATION_BOOST = 1.25

# CLAHE untuk memperbaiki kontras pencahayaan
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# ------------------------------------------------------------------ #
#  PARAMETER AKURASI KEPUTUSAN WARNA                                 #
# ------------------------------------------------------------------ #

# Dibuat sedikit lebih rendah agar uang dengan warna pucat tetap bisa terbaca
PERSENTASE_THRESHOLD = 4.0

# Jika selisih skor terlalu kecil, hasil dianggap tidak yakin
SKOR_DIFF_THRESHOLD = 1.2

# ------------------------------------------------------------------ #
#  PARAMETER KONDISI CAHAYA                                          #
# ------------------------------------------------------------------ #

# Jangan terlalu tinggi, karena nanti kondisi ruangan biasa dianggap gelap
V_LIGHTING_THRESHOLD = 40.0

# ------------------------------------------------------------------ #
#  PARAMETER TEMPLATE MATCHING                                       #
# ------------------------------------------------------------------ #

# Template kuat baru boleh mengalahkan HSV
TEMPLATE_THRESHOLD_KUAT = 0.78

# Template sedang hanya dipakai untuk membantu, bukan langsung menang
TEMPLATE_THRESHOLD_SEDANG = 0.68

# Khusus koreksi HSV Rp2.000 yang sering false positive
TEMPLATE_THRESHOLD_KOREKSI_2000 = 0.72

# ------------------------------------------------------------------ #
#  PARAMETER SUARA                                                   #
# ------------------------------------------------------------------ #

COOLDOWN_SUARA = 2.0

# ------------------------------------------------------------------ #
#  TAMPILAN DEBUG HYBRID                                             #
# ------------------------------------------------------------------ #

TAMPILKAN_INFO_HYBRID = True

# ------------------------------------------------------------------ #
#  DEFINISI RANGE WARNA HSV UNTUK SETIAP NOMINAL RUPIAH              #
# ------------------------------------------------------------------ #
#
# Catatan:
# H = 0–179
# S = 0–255
# V = 0–255
#
# Perbaikan utama:
# - Rp20.000: S minimum diturunkan agar hijau pucat tetap masuk
# - Rp2.000 : S maksimum diperkecil dan bobot diturunkan agar tidak mudah menang
# ------------------------------------------------------------------ #

NOMINAL_HSV = [
    {
        # ---- Rp100.000 — Merah / Pink ----
        "nama"  : "Rp100.000",
        "suara" : "Seratus ribu rupiah",
        "bobot" : 1.25,

        "lower1": np.array([0,   60,  55], dtype=np.uint8),
        "upper1": np.array([12, 255, 255], dtype=np.uint8),

        "lower2": np.array([155, 60,  55], dtype=np.uint8),
        "upper2": np.array([179, 255, 255], dtype=np.uint8),
    },
    {
        # ---- Rp50.000 — Biru ----
        "nama"  : "Rp50.000",
        "suara" : "Lima puluh ribu rupiah",
        "bobot" : 1.05,

        "lower1": np.array([95,  50,  45], dtype=np.uint8),
        "upper1": np.array([132, 255, 255], dtype=np.uint8),

        "lower2": None,
        "upper2": None,
    },
    {
        # ---- Rp20.000 — Hijau ----
        # PERBAIKAN:
        # S minimum diturunkan dari 80 ke 35 agar hijau pucat tetap terdeteksi.
        # Bobot dinaikkan karena Rp20.000 sering kalah oleh Rp2.000 saat redup.
        "nama"  : "Rp20.000",
        "suara" : "Dua puluh ribu rupiah",
        "bobot" : 1.40,

        "lower1": np.array([35,  35,  40], dtype=np.uint8),
        "upper1": np.array([92, 255, 255], dtype=np.uint8),

        "lower2": None,
        "upper2": None,
    },
    {
        # ---- Rp10.000 — Ungu ----
        "nama"  : "Rp10.000",
        "suara" : "Sepuluh ribu rupiah",
        "bobot" : 1.05,

        "lower1": np.array([125, 35,  45], dtype=np.uint8),
        "upper1": np.array([162, 255, 255], dtype=np.uint8),

        "lower2": None,
        "upper2": None,
    },
    {
        # ---- Rp5.000 — Coklat / Kuning Keemasan ----
        "nama"  : "Rp5.000",
        "suara" : "Lima ribu rupiah",
        "bobot" : 1.0,

        "lower1": np.array([13,  45,  45], dtype=np.uint8),
        "upper1": np.array([36, 255, 255], dtype=np.uint8),

        "lower2": None,
        "upper2": None,
    },
    {
        # ---- Rp2.000 — Abu-abu ----
        # PERBAIKAN:
        # Bobot diturunkan besar karena abu-abu sangat mudah false positive.
        # S maksimum diperkecil agar warna uang lain yang redup tidak mudah masuk.
        "nama"  : "Rp2.000",
        "suara" : "Dua ribu rupiah",
        "bobot" : 0.20,

        "lower1": np.array([0,   0,   105], dtype=np.uint8),
        "upper1": np.array([179, 28,  185], dtype=np.uint8),

        "lower2": None,
        "upper2": None,
    },
    {
        # ---- Rp1.000 — Hijau Kebiruan / Teal ----
        "nama"  : "Rp1.000",
        "suara" : "Seribu rupiah",
        "bobot" : 0.90,

        "lower1": np.array([82,  45,  45], dtype=np.uint8),
        "upper1": np.array([102, 255, 255], dtype=np.uint8),

        "lower2": None,
        "upper2": None,
    },
]