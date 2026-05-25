"""
====================================================================
money_validator.py — Validasi Apakah Objek Benar-benar Uang Kertas
====================================================================
Modul ini bertugas memeriksa tiga ciri fisik uang kertas:

  1. validasi_bentuk_uang(contour, frame_area)
     Memeriksa aspek rasio dan rectangularity bounding box kontur.
     Uang kertas cenderung berbentuk persegi panjang yang solid.

  2. validasi_tekstur_uang(roi)
     Memeriksa kepadatan tepi (edge density) pada ROI.
     Uang kertas penuh dengan detail: nomor seri, gambar, teks —
     sehingga edge density-nya cukup tinggi. Benda polos seperti
     tembok atau kertas kosong akan gagal di sini.

  3. validasi_variasi_warna(roi)
     Memeriksa variasi warna melalui standar deviasi Saturation (S)
     dan Value (V) pada ruang warna HSV.
     Uang kertas berwarna-warni dan gradatif → std S dan V cukup tinggi.
     Benda monokrom seperti plastik merah polos akan gagal di sini.

Semua fungsi mengembalikan True (lolos) atau False (gagal).
Tidak ada machine learning — murni analisis geometri dan statistik.
====================================================================
"""

import cv2
import numpy as np


# ------------------------------------------------------------------ #
#  PARAMETER AMBANG BATAS VALIDASI                                   #
# ------------------------------------------------------------------ #

# Batas minimum edge density agar objek dianggap bertekstur cukup
# (seperti uang kertas). Nilai 0.04 artinya minimal 4% piksel adalah tepi.
EDGE_DENSITY_MIN = 0.012

# Batas minimum standar deviasi Saturation (S) pada HSV.
# Uang kertas memiliki variasi saturasi yang cukup tinggi.
STD_SATURATION_MIN = 10.0

# Batas minimum standar deviasi Value (V) pada HSV.
# Uang kertas memiliki variasi kecerahan karena detail gambar dan teks.
STD_VALUE_MIN = 8.0

# Batas minimum rectangularity (luas kontur / luas bounding box).
# Nilai 0.6 artinya minimal 60% bounding box terisi oleh kontur.
# Nilai lebih tinggi = lebih persegi panjang = lebih mirip uang.
RECTANGULARITY_MIN = 0.30


# ------------------------------------------------------------------ #
#  FUNGSI VALIDASI BENTUK                                            #
# ------------------------------------------------------------------ #

def validasi_bentuk_uang(contour: np.ndarray, frame_area: int) -> bool:
    """
    Memeriksa apakah kontur berbentuk seperti uang kertas (persegi panjang).

    Dua syarat harus terpenuhi:
      1. Rectangularity >= RECTANGULARITY_MIN
         → Mengukur seberapa "padat" kontur mengisi bounding box-nya.
         → Uang kertas biasanya hampir memenuhi seluruh bounding box.
         → Rumus: luas_kontur / (lebar_bbox × tinggi_bbox)

      2. Aspek rasio dalam rentang wajar (1.4 – 3.2)
         → Uang kertas selalu lebih panjang dari lebarnya.
         → Dihitung sebagai max(w, h) / min(w, h).

    Parameter:
        contour    : numpy array kontur dari cv2.findContours()
        frame_area : luas total frame kamera (dipakai untuk batas max)

    Mengembalikan:
        True  jika kontur memenuhi syarat bentuk uang
        False jika tidak memenuhi
    """
    # Dapatkan bounding box (kotak terkecil yang membungkus kontur)
    x, y, w, h = cv2.boundingRect(contour)

    # Hindari pembagian nol
    if w == 0 or h == 0:
        return False

    # Hitung luas area bounding box
    luas_bbox = w * h

    # Hitung luas aktual kontur (bukan bounding box, tapi kontur itu sendiri)
    luas_kontur = cv2.contourArea(contour)

    # Rectangularity: seberapa padat kontur mengisi bounding box-nya
    # Nilai mendekati 1.0 berarti hampir persegi panjang sempurna
    rectangularity = luas_kontur / luas_bbox if luas_bbox > 0 else 0

    # Aspek rasio: selalu diambil yang lebih besar dibagi yang lebih kecil
    # supaya orientasi landscape/portrait tidak masalah
    aspek_rasio = max(w, h) / min(w, h)

    # Kedua syarat harus terpenuhi sekaligus
    lolos_rectangularity = rectangularity >= RECTANGULARITY_MIN
    lolos_aspek          = 1.4 <= aspek_rasio <= 3.2

    return lolos_rectangularity and lolos_aspek


# ------------------------------------------------------------------ #
#  FUNGSI VALIDASI TEKSTUR                                           #
# ------------------------------------------------------------------ #

def validasi_tekstur_uang(roi: np.ndarray) -> bool:
    """
    Memeriksa apakah ROI memiliki tekstur yang cukup kompleks seperti uang.

    Cara kerja:
      1. Konversi ROI ke grayscale
      2. Jalankan Canny edge detection
      3. Hitung edge density = jumlah piksel tepi / total piksel
      4. Jika edge density >= EDGE_DENSITY_MIN → lolos

    Uang kertas penuh dengan detail halus (angka seri, ornamen, gambar)
    sehingga menghasilkan banyak tepi. Benda polos (tembok, meja, baju
    polos) menghasilkan sedikit tepi.

    Parameter:
        roi : numpy array BGR (sub-gambar area kandidat uang)

    Mengembalikan:
        True  jika tekstur cukup kompleks (lolos)
        False jika terlalu polos (bukan uang)
    """
    # Pastikan ROI tidak kosong
    if roi is None or roi.size == 0:
        return False

    # Konversi ke grayscale untuk deteksi tepi
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Terapkan Canny edge detection dengan ambang batas sederhana
    # (tidak perlu adaptif di sini karena hanya menghitung kepadatan tepi)
    tepi = cv2.Canny(gray, 50, 150)

    # Hitung total piksel dan piksel yang merupakan tepi (nilai = 255)
    total_piksel = tepi.shape[0] * tepi.shape[1]
    if total_piksel == 0:
        return False

    piksel_tepi  = np.sum(tepi == 255)

    # Edge density: proporsi piksel tepi terhadap total piksel
    edge_density = piksel_tepi / total_piksel

    # Lolos jika edge density cukup tinggi
    return edge_density >= EDGE_DENSITY_MIN


# ------------------------------------------------------------------ #
#  FUNGSI VALIDASI VARIASI WARNA                                     #
# ------------------------------------------------------------------ #

def validasi_variasi_warna(roi: np.ndarray) -> bool:
    """
    Memeriksa apakah ROI memiliki variasi warna yang cukup seperti uang.

    Cara kerja:
      1. Konversi ROI ke HSV
      2. Hitung standar deviasi channel S (Saturation) dan V (Value)
      3. Jika std_s >= STD_SATURATION_MIN DAN std_v >= STD_VALUE_MIN → lolos

    Uang kertas memiliki banyak warna berbeda-beda dan gradasi kecerahan
    karena gambar, ornamen, dan teks. Ini menghasilkan nilai standar deviasi
    S dan V yang cukup tinggi.

    Benda monokrom (plastik biru polos, kertas merah polos) memiliki
    S dan V yang seragam → standar deviasi rendah → gagal validasi.

    Parameter:
        roi : numpy array BGR (sub-gambar area kandidat uang)

    Mengembalikan:
        True  jika variasi warna cukup (lolos)
        False jika terlalu seragam warnanya (bukan uang)
    """
    # Pastikan ROI tidak kosong
    if roi is None or roi.size == 0:
        return False

    # Konversi ke HSV; kita hanya butuh channel S dan V
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Pisahkan channel: H = Hue, S = Saturation, V = Value
    _, s_channel, v_channel = cv2.split(hsv)

    # Hitung standar deviasi masing-masing channel
    # np.std() menghitung seberapa bervariasi nilai piksel dalam gambar
    std_s = float(np.std(s_channel))
    std_v = float(np.std(v_channel))

    # Kedua channel harus melewati ambang batas minimum
    lolos_saturation = std_s >= STD_SATURATION_MIN
    lolos_value      = std_v >= STD_VALUE_MIN

    return lolos_saturation and lolos_value
