"""
====================================================================
config.py — Konfigurasi Global Vision-to-Audio Bridge
====================================================================
Versi perbaikan v2:
  - Range HSV per nominal diperbaiki dan diperketat (bukan lagi catch-all)
  - Rp100.000 merah: range H diperluas dan threshold persentase dinaikkan
  - Rp2.000 abu-abu: bobot sangat rendah agar tidak jadi false positive
  - PERSENTASE_THRESHOLD dinaikkan agar ruang kosong tidak dideteksi
  - SKOR_DIFF_THRESHOLD dinaikkan agar keputusan lebih tegas
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

# Aktifkan validasi agar kotak kosong tidak diproses
VALIDASI_KETAT_GUIDE = True

# ------------------------------------------------------------------ #
#  PARAMETER PREPROCESSING                                           #
# ------------------------------------------------------------------ #

GUNAKAN_PREPROCESSING = True
GAMMA_VALUE = 1.25
SATURATION_BOOST = 1.25
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# ------------------------------------------------------------------ #
#  PARAMETER AKURASI KEPUTUSAN WARNA                                 #
# ------------------------------------------------------------------ #

# Dinaikkan: minimal 12% piksel harus cocok baru dianggap terdeteksi
# Ini mencegah ruang kosong / background terdeteksi sebagai uang
PERSENTASE_THRESHOLD = 12.0

# Dinaikkan: selisih skor harus cukup besar agar tidak ambigu
SKOR_DIFF_THRESHOLD = 5.0

# ------------------------------------------------------------------ #
#  PARAMETER KONDISI CAHAYA                                          #
# ------------------------------------------------------------------ #

V_LIGHTING_THRESHOLD = 40.0

# ------------------------------------------------------------------ #
#  PARAMETER TEMPLATE MATCHING                                       #
# ------------------------------------------------------------------ #

TEMPLATE_THRESHOLD_KUAT = 0.55
TEMPLATE_THRESHOLD_SEDANG = 0.35
TEMPLATE_THRESHOLD_KOREKSI_2000 = 0.45

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
# Catatan penting range HSV OpenCV:
# H = 0–179 (merah ada di 0-10 DAN 160-179)
# S = 0–255 (makin tinggi = makin jenuh/pekat)
# V = 0–255 (makin tinggi = makin terang)
#
# PERBAIKAN UTAMA:
# - Setiap range sekarang SPESIFIK ke warna dominan masing-masing nominal
# - S minimum dinaikkan (>=40) agar abu-abu/warna pucat tidak masuk semua range
# - Range tidak lagi tumpang tindih berlebihan
# ------------------------------------------------------------------ #

NOMINAL_HSV = [
    {
        # ---- Rp100.000 — Merah / Merah Muda ----
        # Uang 100rb didominasi warna merah terang dan pink
        # H: 0-10 (merah bawah) + 160-179 (merah atas)
        # S: minimal 50 agar tidak masuk ke warna kulit/putih
        "nama"  : "Rp100.000",
        "suara" : "Seratus ribu rupiah",
        "bobot" : 1.50,
        "lower1": np.array([0,   50,  50], dtype=np.uint8),
        "upper1": np.array([10, 255, 255], dtype=np.uint8),
        "lower2": np.array([160, 50,  50], dtype=np.uint8),
        "upper2": np.array([179, 255, 255], dtype=np.uint8),
        "ranges": []
    },
    {
        # ---- Rp50.000 — Biru Tua / Biru Dongker ----
        # H: 100-130 (biru murni)
        # S: minimal 60 agar langit/dinding biru tidak masuk
        "nama"  : "Rp50.000",
        "suara" : "Lima puluh ribu rupiah",
        "bobot" : 1.20,
        "lower1": np.array([100, 60,  40], dtype=np.uint8),
        "upper1": np.array([130, 255, 255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "ranges": []
    },
    {
        # ---- Rp20.000 — Hijau ----
        # H: 40-90 (hijau, dari hijau-kuning sampai hijau-biru)
        # S: minimal 50 agar daun/tumbuhan dengan pencahayaan redup tidak masuk
        "nama"  : "Rp20.000",
        "suara" : "Dua puluh ribu rupiah",
        "bobot" : 1.40,
        "lower1": np.array([40,  50,  40], dtype=np.uint8),
        "upper1": np.array([90, 255, 255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "ranges": []
    },
    {
        # ---- Rp10.000 — Ungu / Violet ----
        # H: 130-155 (ungu/violet)
        # S: minimal 40
        "nama"  : "Rp10.000",
        "suara" : "Sepuluh ribu rupiah",
        "bobot" : 1.20,
        "lower1": np.array([130, 40,  40], dtype=np.uint8),
        "upper1": np.array([155, 255, 255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "ranges": []
    },
    {
        # ---- Rp5.000 — Coklat / Orange Tua ----
        # H: 10-25 (orange/coklat)
        # S: minimal 60 agar tidak tumpang tindih dengan kulit
        "nama"  : "Rp5.000",
        "suara" : "Lima ribu rupiah",
        "bobot" : 1.20,
        "lower1": np.array([10,  60,  40], dtype=np.uint8),
        "upper1": np.array([25, 255, 255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "ranges": []
    },
    {
        # ---- Rp2.000 — Abu-abu / Kebiruan Pucat ----
        # Warna abu-abu paling susah: S rendah, V sedang
        # Diberi bobot sangat rendah agar tidak mudah menang
        # Hanya menang kalau nominal lain semua benar-benar tidak ada
        "nama"  : "Rp2.000",
        "suara" : "Dua ribu rupiah",
        "bobot" : 0.60,
        "lower1": np.array([90,  15,  60], dtype=np.uint8),
        "upper1": np.array([130, 80, 200], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "ranges": []
    },
    {
        # ---- Rp1.000 — Hijau Kekuningan / Kuning-Hijau ----
        # H: 25-45 (kuning-hijau, berbeda dari Rp20.000 yang lebih murni hijau)
        # S: minimal 50
        "nama"  : "Rp1.000",
        "suara" : "Seribu rupiah",
        "bobot" : 1.10,
        "lower1": np.array([25,  50,  40], dtype=np.uint8),
        "upper1": np.array([45, 255, 255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "ranges": []
    }
]
