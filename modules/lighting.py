"""
====================================================================
lighting.py — Deteksi Kondisi Pencahayaan dari ROI HSV
====================================================================
Modul ini menganalisis kecerahan rata-rata area ROI untuk menentukan
apakah kondisi pencahayaan cukup untuk melakukan deteksi warna yang
akurat.

Prinsip kerja:
  - Ruang warna HSV memisahkan informasi kecerahan (V = Value) dari
    informasi warna (H = Hue, S = Saturation).
  - Channel V berkisar 0 (hitam penuh) hingga 255 (putih penuh).
  - Jika rata-rata V terlalu rendah, artinya ROI terlalu gelap dan
    deteksi warna akan menghasilkan keluaran tidak akurat.
====================================================================
"""

import numpy as np
from . import config


def cek_kondisi_cahaya(roi_hsv: np.ndarray) -> bool:
    """
    Memeriksa apakah kondisi cahaya pada ROI cukup untuk deteksi warna.

    Fungsi ini mengekstrak channel ke-2 (indeks 2) dari array HSV yang
    merepresentasikan kecerahan (Value), lalu menghitung rata-ratanya
    menggunakan np.mean.

    Parameter:
        roi_hsv : numpy array berformat HSV, shape (H, W, 3).
                  Harus sudah dikonversi dari BGR sebelum dikirim ke sini
                  menggunakan cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).

    Mengembalikan:
        True  → Cahaya KURANG (rata-rata V di bawah V_LIGHTING_THRESHOLD).
                 Loop utama harus membypass deteksi warna dan menampilkan
                 label "Cahaya Kurang".
        False → Cahaya CUKUP. Deteksi warna dapat dilanjutkan secara normal.

    Contoh penggunaan:
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        if cek_kondisi_cahaya(roi_hsv):
            label_aktif = "Cahaya Kurang"
        else:
            label_aktif = tentukan_nominal(roi)
    """

    # Ambil channel V (kecerahan) saja — indeks 2 pada dimensi terakhir
    # roi_hsv[:, :, 0] = Hue
    # roi_hsv[:, :, 1] = Saturation
    # roi_hsv[:, :, 2] = Value (Kecerahan)  ← yang kita butuhkan
    rata_rata_v = np.mean(roi_hsv[:, :, 2])

    # Bandingkan dengan ambang batas yang dikonfigurasi di config.py.
    # Jika kecerahan di bawah threshold → cahaya redup → kembalikan True.
    return rata_rata_v < config.V_LIGHTING_THRESHOLD
