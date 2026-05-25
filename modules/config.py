"""
====================================================================
config.py — Konfigurasi Global Vision-to-Audio Bridge
====================================================================
Semua konstanta numerik dan definisi warna HSV disentralisasi
di sini agar mudah dikalibrasi tanpa menyentuh logika utama.

Versi v5 tambahan:
  - TAMPILKAN_INFO_HYBRID : toggle tampilan debug HSV vs Template
====================================================================
"""

import numpy as np

# ------------------------------------------------------------------ #
#  DIMENSI FRAME KAMERA                                              #
# ------------------------------------------------------------------ #

# Lebar frame yang diambil dari kamera (piksel)
FRAME_WIDTH  = 640

# Tinggi frame yang diambil dari kamera (piksel)
FRAME_HEIGHT = 480

# ------------------------------------------------------------------ #
#  PARAMETER DETEKSI ROI (REGION OF INTEREST)                        #
# ------------------------------------------------------------------ #

# Aspek rasio minimum bounding box kontur uang (lebar / tinggi)
ASPEK_RASIO_MIN = 1.25

# Aspek rasio maksimum; di atas ini dianggap bukan uang
ASPEK_RASIO_MAX = 3.8

# Luas bounding box minimum (piksel persegi)
LUAS_ROI_MIN = 6_000

# Luas bounding box maksimum (piksel persegi)
LUAS_ROI_MAX = 220_000


# ------------------------------------------------------------------ #
#  PARAMETER ROI GUIDE / MANUAL                                      #
# ------------------------------------------------------------------ #

# Rasio posisi kotak panduan terhadap dimensi frame
GUIDE_X_RATIO = 0.08
GUIDE_Y_RATIO = 0.26
GUIDE_W_RATIO = 0.84
GUIDE_H_RATIO = 0.48

# Validasi tekstur/variasi warna di mode GUIDE.
# False = lebih longgar (cocok untuk demo/ujian dengan kamera laptop).
# True  = lebih ketat (cocok jika background dan cahaya sudah terkontrol).
VALIDASI_KETAT_GUIDE = False

# ------------------------------------------------------------------ #
#  PARAMETER AKURASI KEPUTUSAN WARNA                                 #
# ------------------------------------------------------------------ #

# Ambang persentase warna minimum (%)
PERSENTASE_THRESHOLD = 5.0

# Selisih skor minimum antara peringkat-1 dan peringkat-2
SKOR_DIFF_THRESHOLD = 1.5

# ------------------------------------------------------------------ #
#  PARAMETER KONDISI CAHAYA                                          #
# ------------------------------------------------------------------ #

# Rata-rata nilai V (kecerahan) minimum pada HSV agar cahaya dianggap cukup
V_LIGHTING_THRESHOLD = 45.0

# ------------------------------------------------------------------ #
#  PARAMETER SUARA (TEXT-TO-SPEECH)                                  #
# ------------------------------------------------------------------ #

# Jeda minimal (detik) antara dua pembacaan nominal berturut-turut
COOLDOWN_SUARA = 2.0

# ------------------------------------------------------------------ #
#  TAMPILAN DEBUG HYBRID (HSV vs Template)                           #
# ------------------------------------------------------------------ #

# Jika True, tampilkan baris info HSV / Template / Sumber di layar.
# Berguna saat demo ke dosen untuk menjelaskan cara kerja sistem.
# Tekan H saat program berjalan untuk toggle on/off.
TAMPILKAN_INFO_HYBRID = True

# ------------------------------------------------------------------ #
#  DEFINISI RANGE WARNA HSV UNTUK SETIAP NOMINAL RUPIAH             #
# ------------------------------------------------------------------ #
#
# Format setiap entry dictionary:
#   "nama"   : Label teks yang ditampilkan di layar
#   "suara"  : Teks yang dibacakan oleh TTS
#   "bobot"  : Faktor pengali skor
#   "lower1" : Array HSV batas bawah range utama
#   "upper1" : Array HSV batas atas range utama
#   "lower2" : Array HSV batas bawah range kedua (opsional, untuk merah)
#   "upper2" : Array HSV batas atas range kedua (opsional, untuk merah)
#
# Catatan teknis HSV di OpenCV:
#   H  : 0 – 179
#   S  : 0 – 255
#   V  : 0 – 255
#
# Merah memerlukan DUA range karena H merah "melingkari" 0°/180°.

NOMINAL_HSV = [
    {
        # ---- Rp100.000 — Dominan Merah/Pink ----
        "nama"  : "Rp100.000",
        "suara" : "Seratus ribu rupiah",
        "bobot" : 1.3,
        "lower1": np.array([0,   80,  80],  dtype=np.uint8),
        "upper1": np.array([10,  255, 255], dtype=np.uint8),
        "lower2": np.array([160, 80,  80],  dtype=np.uint8),
        "upper2": np.array([179, 255, 255], dtype=np.uint8),
    },
    {
        # ---- Rp50.000 — Dominan Biru ----
        "nama"  : "Rp50.000",
        "suara" : "Lima puluh ribu rupiah",
        "bobot" : 1.0,
        "lower1": np.array([100, 80,  80],  dtype=np.uint8),
        "upper1": np.array([130, 255, 255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
    },
    {
        # ---- Rp20.000 — Dominan Hijau ----
        "nama"  : "Rp20.000",
        "suara" : "Dua puluh ribu rupiah",
        "bobot" : 1.0,
        "lower1": np.array([36,  80,  80],  dtype=np.uint8),
        "upper1": np.array([85,  255, 255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
    },
    {
        # ---- Rp10.000 — Dominan Ungu ----
        "nama"  : "Rp10.000",
        "suara" : "Sepuluh ribu rupiah",
        "bobot" : 1.0,
        "lower1": np.array([130, 50,  50],  dtype=np.uint8),
        "upper1": np.array([160, 255, 255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
    },
    {
        # ---- Rp5.000 — Dominan Coklat/Kuning Keemasan ----
        "nama"  : "Rp5.000",
        "suara" : "Lima ribu rupiah",
        "bobot" : 1.0,
        "lower1": np.array([15,  80,  80],  dtype=np.uint8),
        "upper1": np.array([35,  255, 255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
    },
    {
        # ---- Rp2.000 — Dominan Abu-abu ----
        "nama"  : "Rp2.000",
        "suara" : "Dua ribu rupiah",
        # Bobot sangat rendah karena abu-abu sangat mudah false positive
        "bobot" : 0.35,
        "lower1": np.array([0,   0,   100], dtype=np.uint8),
        "upper1": np.array([179, 35,  190], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
    },
    {
        # ---- Rp1.000 — Dominan Hijau Kebiruan (Teal/Cyan) ----
        "nama"  : "Rp1.000",
        "suara" : "Seribu rupiah",
        "bobot" : 0.9,
        "lower1": np.array([85,  80,  80],  dtype=np.uint8),
        "upper1": np.array([100, 255, 255], dtype=np.uint8),
        "lower2": None,
        "upper2": None,
    },
]
