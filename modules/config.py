"""
====================================================================
config.py — Konfigurasi Global Vision-to-Audio Bridge
====================================================================
Versi perbaikan v3:
  - HSV range diperluas untuk uang fisik nyata (pucat/kusam)
  - Range digital tetap dipertahankan sebagai primary
  - Tambah range fisik sebagai secondary untuk toleransi lebih besar
  - Threshold disesuaikan agar false positive tetap terjaga
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

# Diturunkan sedikit dari 12 -> 10 agar uang fisik yang pucat
# tetap lolos threshold (warna fisik lebih sedikit piksel cocok)
PERSENTASE_THRESHOLD = 10.0

# Selisih skor harus cukup besar agar tidak ambigu
SKOR_DIFF_THRESHOLD = 4.0

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
#  PARAMETER CAPTURE & VALIDASI AUDIO                                #
# ------------------------------------------------------------------ #

# Jumlah frame yang dikumpulkan untuk validasi sebelum output audio
CAPTURE_WINDOW_SIZE = 12

# Minimal persentase frame yang harus setuju (majority vote)
CAPTURE_CONSENSUS_PERSEN = 55.0

# Cooldown setelah audio berhasil diputar (detik)
CAPTURE_COOLDOWN = 3.0

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
# PERBAIKAN v3:
# - Setiap nominal sekarang punya range UTAMA (digital) + range FISIK
# - Range fisik: S minimum diturunkan, V minimum diturunkan
# - Ini menangkap uang fisik yang pucat/kusam tanpa mengorbankan presisi digital
# - Range tidak tumpang tindih berlebihan antar nominal
# ------------------------------------------------------------------ #

NOMINAL_HSV = [
    {
        # ---- Rp100.000 — Merah / Merah Muda ----
        # Uang 100rb didominasi warna merah terang dan pink
        # Digital: H 0-10 + 160-179, S>=50
        # Fisik: range lebih lebar, S>=25, V lebih rendah
        "nama"  : "Rp100.000",
        "suara" : "Seratus ribu rupiah",
        "bobot" : 1.50,
        "lower1": np.array([0,   40,  40], dtype=np.uint8),
        "upper1": np.array([12, 255, 255], dtype=np.uint8),
        "lower2": np.array([155, 40,  40], dtype=np.uint8),
        "upper2": np.array([179, 255, 255], dtype=np.uint8),
        "ranges": []
    },
    {
        # ---- Rp50.000 — Biru Tua / Biru Dongker ----
        # Digital: H 100-130, S>=60
        # Fisik: S lebih rendah (biru pucat), V lebih rendah (gelap)
        "nama"  : "Rp50.000",
        "suara" : "Lima puluh ribu rupiah",
        "bobot" : 1.20,
        "lower1": np.array([95,  40,  30], dtype=np.uint8),
        "upper1": np.array([135, 255, 255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "ranges": []
    },
    {
        # ---- Rp20.000 — Hijau ----
        # Digital: H 40-90, S>=50
        # Fisik: hijau yang kusam, S lebih rendah
        "nama"  : "Rp20.000",
        "suara" : "Dua puluh ribu rupiah",
        "bobot" : 1.40,
        "lower1": np.array([35,  35,  35], dtype=np.uint8),
        "upper1": np.array([90, 255, 255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "ranges": []
    },
    {
        # ---- Rp10.000 — Ungu / Violet ----
        # Digital: H 130-155, S>=40
        # Fisik: ungu pucat, range H lebih lebar
        "nama"  : "Rp10.000",
        "suara" : "Sepuluh ribu rupiah",
        "bobot" : 1.20,
        "lower1": np.array([125, 30,  35], dtype=np.uint8),
        "upper1": np.array([158, 255, 255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "ranges": []
    },
    {
        # ---- Rp5.000 — Coklat / Orange Tua ----
        # Digital: H 10-25, S>=60
        # Fisik: coklat kusam, S lebih rendah
        "nama"  : "Rp5.000",
        "suara" : "Lima ribu rupiah",
        "bobot" : 1.20,
        "lower1": np.array([8,   40,  35], dtype=np.uint8),
        "upper1": np.array([28, 255, 255], dtype=np.uint8),
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
        "lower1": np.array([85,  12,  50], dtype=np.uint8),
        "upper1": np.array([135, 85, 210], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "ranges": []
    },
    {
        # ---- Rp1.000 — Hijau Kekuningan / Kuning-Hijau ----
        # Digital: H 25-45, S>=50
        # Fisik: kuning-hijau pucat
        "nama"  : "Rp1.000",
        "suara" : "Seribu rupiah",
        "bobot" : 1.10,
        "lower1": np.array([22,  35,  35], dtype=np.uint8),
        "upper1": np.array([48, 255, 255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
        "ranges": []
    }
]
