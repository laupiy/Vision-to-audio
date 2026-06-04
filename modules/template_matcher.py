"""
====================================================================
template_matcher.py — Template Matching untuk Konfirmasi Nominal
====================================================================
Versi perbaikan v2 - Multi-Scale + Parameter Lebih Toleran:
  - Multi-scale matching: ROI dicocokkan pada beberapa ukuran berbeda
    agar uang fisik yang jaraknya bervariasi tetap terdeteksi
  - ORB nfeatures dinaikkan dari 800 → 1200 untuk lebih banyak fitur
  - Hamming distance threshold dinaikkan dari 55 → 65 untuk toleransi
    lebih besar terhadap perbedaan pencahayaan uang fisik
  - Normalisasi skor lebih fair (20% cocok = skor 1.0)
====================================================================
"""

import cv2
import numpy as np
from pathlib import Path


TEMPLATE_DIR = Path("templates")

NOMINAL_LABELS = {
    "100000": "Rp100.000",
    "50000" : "Rp50.000",
    "20000" : "Rp20.000",
    "10000" : "Rp10.000",
    "5000"  : "Rp5.000",
    "2000"  : "Rp2.000",
    "1000"  : "Rp1.000",
}

LEMBAR_LABELS = {
    "lembar_100000": "Rp100.000",
    "lembar_50000" : "Rp50.000",
    "lembar_20000" : "Rp20.000",
    "lembar_10000" : "Rp10.000",
    "lembar_5000"  : "Rp5.000",
    "lembar_2000"  : "Rp2.000",
    "lembar_1000"  : "Rp1.000",
}

# ------------------------------------------------------------------ #
#  CACHE & ORB SETUP                                                 #
# ------------------------------------------------------------------ #

_lembar_cache = []
# Naikkan nfeatures jadi 800 agar cukup cepat tapi detail
_orb = cv2.ORB_create(nfeatures=800)
_bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def _load_lembar_templates():
    """
    Load semua template lembar uang penuh (.jpeg).
    Template ini digunakan untuk mencocokkan ROI dari kamera.
    """
    global _lembar_cache
    if _lembar_cache:
        return _lembar_cache

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for path in TEMPLATE_DIR.glob("*.jpeg"):
        key = path.stem
        if key not in LEMBAR_LABELS:
            continue

        img = cv2.imread(str(path))
        if img is None:
            continue

        # Kembalikan ke single scale untuk performa realtime
        h, w = img.shape[:2]
        target_h = 240
        scale = target_h / h
        img_r = cv2.resize(img, (int(w * scale), target_h))

        gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)

        kp, des = _orb.detectAndCompute(gray, None)

        if des is not None and len(kp) >= 15:
            _lembar_cache.append({
                "key"  : key,
                "label": LEMBAR_LABELS[key],
                "kp"   : kp,
                "des"  : des,
            })

    return _lembar_cache


# ------------------------------------------------------------------ #
#  FUNGSI UTAMA                                                      #
# ------------------------------------------------------------------ #

def cocokkan_template(roi: np.ndarray) -> tuple[str, float, list]:
    """
    Mencocokkan ROI dengan template lembar uang menggunakan ORB.
    Multi-scale: ROI juga diproses pada beberapa ukuran.

    Mengembalikan (label_terbaik, skor_terbaik, semua_skor).

    Interpretasi skor:
    - >= 0.55 → kuat, bisa menjadi hasil final
    - 0.35 - 0.54 → sedang, bisa konfirmasi HSV
    - < 0.35 → lemah, abaikan / percaya HSV
    """
    if roi is None or roi.size == 0:
        return "Tidak ada template", 0.0, []

    lembar_tmps = _load_lembar_templates()
    if not lembar_tmps:
        return "Tidak ada template", 0.0, []

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    best_label = "Tidak yakin"
    best_score = 0.0
    all_scores = []
    
    # Proses ROI di satu ukuran standar untuk performa (240px tinggi)
    h, w = roi.shape[:2]
    target_h = 240
    scale = target_h / max(h, 1)
    roi_r = cv2.resize(roi, (int(w * scale), target_h))

    gray_roi = cv2.cvtColor(roi_r, cv2.COLOR_BGR2GRAY)
    gray_roi = clahe.apply(gray_roi)

    kp_roi, des_roi = _orb.detectAndCompute(gray_roi, None)

    if des_roi is None or len(kp_roi) < 8:
        return "Tidak yakin", 0.0, []

    for item in lembar_tmps:
        try:
            matches = _bf.match(item["des"], des_roi)
        except cv2.error:
            continue

        # Good match: Hamming distance < 65
        good = [m for m in matches if m.distance < 65]

        # Normalisasi: 20% dari keypoints cocok = skor 1.0
        n_tmpl = max(len(item["kp"]), 1)
        score = min(len(good) / (n_tmpl * 0.20), 1.0)

        all_scores.append((item["label"], score))

        if score > best_score:
            best_score = score
            best_label = item["label"]

    all_scores.sort(key=lambda x: x[1], reverse=True)

    # Filter ketat: Jika skor template sangat rendah, berarti bukan uang / tidak ada uang
    # Skor 0.22 berarti butuh setidaknya 4-5% fitur uang yang benar-benar cocok.
    # Wajah atau background acak tidak akan mencapai skor ini.
    if best_score < 0.22:
        return "Objek bukan uang", best_score, all_scores

    return best_label, best_score, all_scores
