"""
====================================================================
color_detector.py — Identifikasi Nominal Uang Berdasarkan Warna HSV
====================================================================
Modul ini mengandung dua fungsi utama:

  1. hitung_skor_warna(hsv_roi, nominal)
     Menghitung persentase piksel ROI yang cocok dengan range warna
     sebuah nominal, lalu dikalikan bobot penalti/bonus.

  2. tentukan_nominal(roi)
     Menerima ROI BGR, menghitung skor semua nominal, menerapkan
     logika keputusan cerdas (threshold + anti-ambiguitas), lalu
     mengembalikan label hasil deteksi.

Logika Keputusan Cerdas:
  - Nominal peringkat-1 VALID hanya jika:
      a) Persentasenya >= PERSENTASE_THRESHOLD (ada warna yang cukup dominan)
      b) Selisih skor peringkat-1 dan peringkat-2 >= SKOR_DIFF_THRESHOLD
         (pemenang unggul jauh, bukan sekadar kebetulan)
  - Jika (a) gagal → "Tidak terdeteksi"
  - Jika (b) gagal → "Tidak yakin"
====================================================================
"""

import cv2
import numpy as np
from . import config


def hitung_skor_warna(hsv_roi: np.ndarray, nominal: dict) -> tuple[float, float]:
    """
    Menghitung skor kecocokan warna antara ROI dengan sebuah nominal.

    Skor dihitung sebagai persentase piksel yang cocok terhadap total
    piksel ROI, kemudian dikalikan bobot dari config (bonus/penalti).

    Parameter:
        hsv_roi : numpy array HSV hasil konversi ROI kamera
        nominal : dictionary nominal dari config.NOMINAL_HSV

    Mengembalikan:
        persentase : float — % piksel cocok sebelum pembobotan (0–100)
        skor_bobot : float — persentase × bobot (yang digunakan untuk ranking)
    """
    # Buat mask piksel yang cocok range warna utama
    mask = cv2.inRange(hsv_roi, nominal["lower1"], nominal["upper1"])

    # Jika ada range warna kedua (khusus merah), gabungkan dengan OR
    if nominal["lower2"] is not None:
        mask2 = cv2.inRange(hsv_roi, nominal["lower2"], nominal["upper2"])
        mask  = cv2.bitwise_or(mask, mask2)

    # Hitung total piksel ROI; hindari pembagian nol jika ROI kosong
    total_piksel = hsv_roi.shape[0] * hsv_roi.shape[1]
    if total_piksel == 0:
        return 0.0, 0.0

    # Piksel yang cocok adalah semua piksel bernilai 255 pada mask
    piksel_cocok = int(np.sum(mask == 255))

    # Persentase kecocokan murni (belum diberi bobot)
    persentase = (piksel_cocok / total_piksel) * 100.0

    # Skor berbobot: dikalikan faktor dari config (bonus/penalti per nominal)
    skor_bobot = persentase * nominal["bobot"]

    return persentase, skor_bobot


def tentukan_nominal(roi: np.ndarray) -> str:
    """
    Menentukan nominal uang yang paling mungkin dari sebuah ROI BGR.

    Alur logika:
        1. Konversi ROI ke HSV
        2. Hitung skor warna untuk setiap nominal
        3. Urutkan nominal dari skor tertinggi ke terendah
        4. Periksa persentase pemenang >= PERSENTASE_THRESHOLD
        5. Periksa selisih skor pemenang dan runner-up >= SKOR_DIFF_THRESHOLD
        6. Kembalikan nama nominal, "Tidak terdeteksi", atau "Tidak yakin"

    Parameter:
        roi : numpy array BGR (hasil crop dari roi_detector.py)

    Mengembalikan:
        str — salah satu dari:
            • "Rp100.000" / "Rp50.000" / ... (nominal terdeteksi)
            • "Tidak terdeteksi" (warna tidak cukup dominan)
            • "Tidak yakin"      (dua nominal terlalu mirip skornya)
    """

    # Pastikan ROI tidak kosong sebelum diproses
    if roi is None or roi.size == 0:
        return "Tidak terdeteksi"

    # ---- Langkah 1: Konversi BGR → HSV ----------------------------------- #
    # HSV lebih stabil terhadap variasi pencahayaan dibanding BGR murni.
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ---- Langkah 2: Hitung skor setiap nominal ---------------------------- #
    hasil_skor = []   # List of (nama_nominal, persentase, skor_bobot)

    for nominal in config.NOMINAL_HSV:
        persentase, skor_bobot = hitung_skor_warna(hsv_roi, nominal)
        hasil_skor.append({
            "nama"       : nominal["nama"],
            "suara"      : nominal["suara"],
            "persentase" : persentase,
            "skor"       : skor_bobot,
        })

    # ---- Langkah 3: Urutkan dari skor tertinggi ke terendah -------------- #
    hasil_skor.sort(key=lambda x: x["skor"], reverse=True)

    # Ambil peringkat 1 (pemenang) dan peringkat 2 (runner-up)
    peringkat_1 = hasil_skor[0]
    peringkat_2 = hasil_skor[1] if len(hasil_skor) > 1 else {"skor": 0.0}

    # ---- Langkah 4: Cek threshold persentase ----------------------------- #
    # Jika pemenang tidak mencapai persentase minimum, tidak ada warna
    # yang cukup dominan → tidak ada uang yang terdeteksi.
    if peringkat_1["persentase"] < config.PERSENTASE_THRESHOLD:
        return "Tidak terdeteksi"

    # ---- Langkah 5: Cek selisih skor (anti-ambiguitas) ------------------- #
    # Jika skor pemenang dan runner-up terlalu berdekatan, deteksi tidak
    # bisa dipercaya — mungkin warna sedang ambigu atau pencahayaan tidak
    # sempurna.
    selisih_skor = peringkat_1["skor"] - peringkat_2["skor"]
    if selisih_skor < config.SKOR_DIFF_THRESHOLD:
        return "Tidak yakin"

    # ---- Langkah 6: Kembalikan nama nominal pemenang --------------------- #
    return peringkat_1["nama"]


def ambil_label_suara(nama_nominal: str) -> str:
    """
    Mengambil string teks suara yang sesuai dengan nama nominal.

    Digunakan oleh speech.py untuk mendapatkan kalimat yang dibacakan TTS,
    misal "Seratus ribu rupiah" dari "Rp100.000".

    Parameter:
        nama_nominal : string nama nominal (misal "Rp100.000")

    Mengembalikan:
        String teks suara; jika tidak ditemukan, kembalikan nama_nominal.
    """
    for nominal in config.NOMINAL_HSV:
        if nominal["nama"] == nama_nominal:
            return nominal["suara"]
    return nama_nominal   # fallback: bacakan nama apa adanya
