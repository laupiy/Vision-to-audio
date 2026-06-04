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
    # Buat mask kosong terlebih dahulu
    mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)

    # Tambahkan range warna dari dataset
    for lower, upper in nominal.get("ranges", []):
        current_mask = cv2.inRange(hsv_roi, lower, upper)
        mask = cv2.bitwise_or(mask, current_mask)

    # Tambahkan range manual jika ada
    if "lower1" in nominal and nominal["lower1"] is not None:
        mask1 = cv2.inRange(hsv_roi, nominal["lower1"], nominal["upper1"])
        mask = cv2.bitwise_or(mask, mask1)

    # Jika ada range warna kedua (khusus merah), gabungkan dengan OR
    if "lower2" in nominal and nominal["lower2"] is not None:
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
    Dioptimalkan dengan Filter Anti-Ambiguitas Khusus 20rb, 50rb, dan 100rb.

    Alur logika:
        1. Konversi ROI ke HSV
        2. Hitung skor warna untuk setiap nominal
        3. Terapkan Penalti Lintas Warna (Cross-Color Penalty) untuk menekan false positive
        4. Urutkan nominal dari skor tertinggi ke terendah
        5. Periksa persentase pemenang >= PERSENTASE_THRESHOLD
        6. Periksa selisih skor pemenang dan runner-up >= SKOR_DIFF_THRESHOLD
        7. Kembalikan nama nominal, "Tidak terdeteksi", atau "Tidak yakin"

    Parameter:
        roi : numpy array BGR (hasil crop dari roi_detector.py)

    Mengembalikan:
        str — salah satu dari nominal terdeteksi, "Tidak terdeteksi", atau "Tidak yakin"
    """

    # Pastikan ROI tidak kosong sebelum diproses
    if roi is None or roi.size == 0:
        return "Tidak terdeteksi"

    # ---- Langkah 1: Konversi BGR → HSV ----------------------------------- #
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ---- Langkah 2: Hitung skor setiap nominal ---------------------------- #
    hasil_skor = []
    mentah = {}  # Menyimpan persentase murni untuk analisis lintas warna

    for nominal in config.NOMINAL_HSV:
        persentase, skor_bobot = hitung_skor_warna(hsv_roi, nominal)
        mentah[nominal["nama"]] = persentase
        hasil_skor.append({
            "nama"       : nominal["nama"],
            "suara"      : nominal["suara"],
            "persentase" : persentase,
            "skor"       : skor_bobot,
        })

    # ---- LANGKAH 3: FILTER PENALTI LINTAS WARNA (ANTI-OVERLAP) ------------ #
    # Mengatasi perebutan skor dominan antara pecahan yang warnanya mirip
    for item in hasil_skor:
        # A. Reduksi Ambiguitas 20rb vs 50rb (Hijau vs Biru/Teal)
        if item["nama"] == "Rp50.000" and mentah.get("Rp20.000", 0) > 10.0:
            item["skor"] *= 0.65  # Beri penalti 35% pada warna biru
            
        if item["nama"] == "Rp20.000" and mentah.get("Rp50.000", 0) > 15.0:
            item["skor"] *= 0.70  # Beri penalti 30% pada warna hijau

        # B. Reduksi Ambiguitas 100rb — threshold diturunkan dari 22% ke 18%
        #    karena uang fisik warna merahnya lebih pucat
        if item["nama"] == "Rp100.000" and item["persentase"] < 18.0:
            item["skor"] *= 0.40  # Beri penalti besar jika warna merah terlalu sedikit

        # C. Reduksi Ambiguitas 5rb vs 1rb (Coklat/Orange vs Kuning-Hijau)
        #    Range H mereka berdekatan (8-28 vs 22-48), bisa tumpang tindih
        if item["nama"] == "Rp5.000" and mentah.get("Rp1.000", 0) > 15.0:
            item["skor"] *= 0.75
        if item["nama"] == "Rp1.000" and mentah.get("Rp5.000", 0) > 15.0:
            item["skor"] *= 0.75

        # D. Reduksi Ambiguitas 10rb vs 100rb (Ungu vs Merah)
        #    Range H 100rb (155-179) berdekatan dengan 10rb (125-158)
        if item["nama"] == "Rp10.000" and mentah.get("Rp100.000", 0) > 15.0:
            item["skor"] *= 0.70
        if item["nama"] == "Rp100.000" and mentah.get("Rp10.000", 0) > 20.0:
            item["skor"] *= 0.75

        # E. Reduksi Ambiguitas 20rb vs 1rb (Hijau vs Kuning-Hijau)
        if item["nama"] == "Rp20.000" and mentah.get("Rp1.000", 0) > 12.0:
            item["skor"] *= 0.70
        if item["nama"] == "Rp1.000" and mentah.get("Rp20.000", 0) > 12.0:
            item["skor"] *= 0.70

    # ---- Langkah 4: Urutkan dari skor tertinggi ke terendah -------------- #
    hasil_skor.sort(key=lambda x: x["skor"], reverse=True)

    # Ambil peringkat 1 (pemenang) dan peringkat 2 (runner-up)
    peringkat_1 = hasil_skor[0]
    peringkat_2 = hasil_skor[1] if len(hasil_skor) > 1 else {"skor": 0.0}

    # ---- Langkah 5: Cek threshold persentase ----------------------------- #
    if peringkat_1["persentase"] < config.PERSENTASE_THRESHOLD:
        return "Tidak terdeteksi"

    # ---- Langkah 6: Cek selisih skor (anti-ambiguitas) ------------------- #
    selisih_skor = peringkat_1["skor"] - peringkat_2["skor"]
    if selisih_skor < config.SKOR_DIFF_THRESHOLD:
        # Jika skornya setelah dipenalti masih bersaing ketat, kembalikan "Tidak yakin" 
        # agar keputusan diserahkan sepenuhnya ke modul Template Matching Anda!
        return "Tidak yakin"

    # ---- Langkah 7: Kembalikan nama nominal pemenang --------------------- #
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
    return nama_nominal  # fallback: bacakan nama apa adanya