"""
====================================================================
template_matcher.py — Template Matching untuk Konfirmasi Nominal
====================================================================
Strategi:
  - Template lembar (*.jpeg): digunakan untuk VALIDASI dan KONFIRMASI
    Cocokkan ROI vs semua lembar uang, ambil yang paling mirip.
    Karena ORB feature matching invariant terhadap skala & rotasi,
    ini adalah cara terbaik dengan dataset yang ada.

  - Template angka (*.jpg/*.png): TIDAK dipakai untuk deteksi nominal
    karena terlalu kecil & tidak cukup fitur unik untuk ORB.
    Disimpan untuk referensi saja.

Catatan penting:
  - Deteksi nominal UTAMA tetap oleh color_detector (HSV).
  - Template matching di sini hanya sebagai KONFIRMASI / TIE-BREAKER.
  - Jika skor template rendah, bukan berarti salah — bisa karena
    pencahayaan atau sudut kamera. HSV tetap diprioritaskan.
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

        # Resize ke tinggi standar untuk konsistensi matching
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

    # Preprocessing ROI: resize dan normalize
    h, w = roi.shape[:2]
    target_h = 200
    scale = target_h / max(h, 1)
    roi_r = cv2.resize(roi, (int(w * scale), target_h))

    gray_roi = cv2.cvtColor(roi_r, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_roi = clahe.apply(gray_roi)

    kp_roi, des_roi = _orb.detectAndCompute(gray_roi, None)

    if des_roi is None or len(kp_roi) < 10:
        return "Tidak yakin", 0.0, []

    best_label = "Tidak yakin"
    best_score = 0.0
    all_scores = []

    for item in lembar_tmps:
        try:
            matches = _bf.match(item["des"], des_roi)
        except cv2.error:
            continue

        # Good match: Hamming distance < 55
        good = [m for m in matches if m.distance < 55]

        # Normalisasi: relatif terhadap keypoints template
        # 30% dari keypoints cocok = skor 1.0
        n_tmpl = max(len(item["kp"]), 1)
        score = min(len(good) / (n_tmpl * 0.30), 1.0)

        all_scores.append((item["label"], score))

        if score > best_score:
            best_score = score
            best_label = item["label"]

    all_scores.sort(key=lambda x: x[1], reverse=True)
    return best_label, best_score, all_scores
