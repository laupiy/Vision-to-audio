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
        "lower1": np.array([0,   9,  64], dtype=np.uint8),
        "upper1": np.array([12, 255, 255], dtype=np.uint8),
        "lower2": np.array([155, 60,  55], dtype=np.uint8),
        "upper2": np.array([179, 255, 255], dtype=np.uint8),
        "ranges": [
            (np.array([0, 9, 93], dtype=np.uint8), np.array([179, 153, 248], dtype=np.uint8)),
            (np.array([0, 11, 70], dtype=np.uint8), np.array([179, 140, 221], dtype=np.uint8)),
            (np.array([0, 10, 64], dtype=np.uint8), np.array([179, 142, 232], dtype=np.uint8))
        ]
    },
    {
        # ---- Rp50.000 — Biru ----
        "nama"  : "Rp50.000",
        "suara" : "Lima puluh ribu rupiah",
        "bobot" : 1.05,
        "lower1": np.array([4,  18,  43], dtype=np.uint8),
        "upper1": np.array([129, 153, 207], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "ranges": [
            (np.array([5, 19, 45], dtype=np.uint8), np.array([129, 156, 223], dtype=np.uint8)),
            (np.array([4, 18, 47], dtype=np.uint8), np.array([167, 155, 232], dtype=np.uint8)),
            (np.array([6, 19, 43], dtype=np.uint8), np.array([150, 153, 207], dtype=np.uint8)),
            (np.array([0, 2, 89], dtype=np.uint8), np.array([168, 121, 237], dtype=np.uint8)),
            (np.array([0, 2, 78], dtype=np.uint8), np.array([176, 94, 237], dtype=np.uint8)),
            (np.array([1, 2, 103], dtype=np.uint8), np.array([173, 88, 237], dtype=np.uint8)),
            (np.array([0, 2, 76], dtype=np.uint8), np.array([173, 123, 237], dtype=np.uint8)),
            (np.array([1, 2, 77], dtype=np.uint8), np.array([171, 107, 237], dtype=np.uint8)),
            (np.array([9, 3, 98], dtype=np.uint8), np.array([135, 123, 178], dtype=np.uint8)),
            (np.array([5, 3, 107], dtype=np.uint8), np.array([150, 104, 164], dtype=np.uint8))
        ]
    },
    {
        # ---- Rp20.000 — Hijau ----
        "nama"  : "Rp20.000",
        "suara" : "Dua puluh ribu rupiah",
        "bobot" : 1.40,
        "lower1": np.array([0,  5,  69], dtype=np.uint8),
        "upper1": np.array([175, 96, 234], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "ranges": [
            (np.array([24, 4, 69], dtype=np.uint8), np.array([170, 98, 242], dtype=np.uint8)),
            (np.array([0, 5, 83], dtype=np.uint8), np.array([178, 96, 243], dtype=np.uint8)),
            (np.array([3, 4, 93], dtype=np.uint8), np.array([175, 96, 234], dtype=np.uint8))
        ]
    },
    {
        # ---- Rp10.000 — Ungu ----
        "nama"  : "Rp10.000",
        "suara" : "Sepuluh ribu rupiah",
        "bobot" : 1.05,
        "lower1": np.array([4, 5,  53], dtype=np.uint8),
        "upper1": np.array([165, 82, 203], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "ranges": [
            (np.array([19, 5, 53], dtype=np.uint8), np.array([166, 96, 212], dtype=np.uint8)),
            (np.array([11, 5, 54], dtype=np.uint8), np.array([165, 82, 223], dtype=np.uint8)),
            (np.array([4, 6, 93], dtype=np.uint8), np.array([174, 88, 203], dtype=np.uint8)),
            (np.array([2, 2, 107], dtype=np.uint8), np.array([171, 47, 237], dtype=np.uint8)),
            (np.array([3, 2, 59], dtype=np.uint8), np.array([173, 66, 237], dtype=np.uint8)),
            (np.array([0, 1, 94], dtype=np.uint8), np.array([170, 82, 237], dtype=np.uint8)),
            (np.array([14, 5, 93], dtype=np.uint8), np.array([170, 43, 169], dtype=np.uint8)),
            (np.array([5, 2, 108], dtype=np.uint8), np.array([173, 81, 237], dtype=np.uint8))
        ]
    },
    {
        # ---- Rp5.000 — Coklat / Kuning Keemasan ----
        "nama"  : "Rp5.000",
        "suara" : "Lima ribu rupiah",
        "bobot" : 1.0,
        "lower1": np.array([0,   8,   62], dtype=np.uint8),
        "upper1": np.array([173, 73, 207], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "ranges": [
            (np.array([1, 11, 62], dtype=np.uint8), np.array([178, 89, 218], dtype=np.uint8)),
            (np.array([1, 8, 73], dtype=np.uint8), np.array([176, 73, 218], dtype=np.uint8)),
            (np.array([0, 10, 85], dtype=np.uint8), np.array([178, 75, 207], dtype=np.uint8)),
            (np.array([0, 5, 52], dtype=np.uint8), np.array([173, 104, 184], dtype=np.uint8)),
            (np.array([0, 8, 92], dtype=np.uint8), np.array([178, 79, 170], dtype=np.uint8)),
            (np.array([2, 2, 102], dtype=np.uint8), np.array([173, 92, 237], dtype=np.uint8)),
            (np.array([0, 5, 100], dtype=np.uint8), np.array([177, 62, 187], dtype=np.uint8)),
            (np.array([0, 4, 57], dtype=np.uint8), np.array([175, 97, 142], dtype=np.uint8))
        ]
    },
    {
        # ---- Rp2.000 — Abu-abu ----
        "nama"  : "Rp2.000",
        "suara" : "Dua ribu rupiah",
        "bobot" : 0.20,
        "lower1": np.array([3,   2,   36], dtype=np.uint8),
        "upper1": np.array([170, 92,  223], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "ranges": [
            (np.array([8, 3, 36], dtype=np.uint8), np.array([170, 100, 230], dtype=np.uint8)),
            (np.array([6, 6, 43], dtype=np.uint8), np.array([170, 93, 223], dtype=np.uint8)),
            (np.array([3, 2, 82], dtype=np.uint8), np.array([170, 92, 225], dtype=np.uint8)),
            (np.array([13, 6, 84], dtype=np.uint8), np.array([160, 80, 165], dtype=np.uint8)),
            (np.array([13, 6, 82], dtype=np.uint8), np.array([158, 76, 162], dtype=np.uint8)),
            (np.array([83, 9, 63], dtype=np.uint8), np.array([163, 78, 141], dtype=np.uint8)),
            (np.array([90, 4, 63], dtype=np.uint8), np.array([163, 75, 184], dtype=np.uint8)),
            (np.array([10, 5, 69], dtype=np.uint8), np.array([146, 94, 207], dtype=np.uint8)),
            (np.array([8, 2, 75], dtype=np.uint8), np.array([158, 83, 237], dtype=np.uint8))
        ]
    },
    {
        # ---- Rp1.000 — Hijau Kebiruan / Teal ----
        "nama"  : "Rp1.000",
        "suara" : "Seribu rupiah",
        "bobot" : 0.90,
        "lower1": np.array([2,  4,  33], dtype=np.uint8),
        "upper1": np.array([173, 77, 221], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "ranges": [
            (np.array([2, 4, 42], dtype=np.uint8), np.array([175, 92, 224], dtype=np.uint8)),
            (np.array([4, 6, 33], dtype=np.uint8), np.array([173, 105, 230], dtype=np.uint8)),
            (np.array([3, 6, 99], dtype=np.uint8), np.array([173, 77, 221], dtype=np.uint8))
        ]
    },
]