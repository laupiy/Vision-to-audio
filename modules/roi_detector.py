"""
====================================================================
roi_detector.py — Deteksi Region of Interest (ROI) Uang Kertas
====================================================================
Versi v3 lebih toleran:
- Tidak memaksa kontur harus persegi panjang sempurna.
- Menggunakan minAreaRect agar uang yang miring tetap bisa terdeteksi.
- Filter bentuk dibuat bertahap: area, aspek rasio, rectangularity.
- Jika kontur valid ditemukan, ROI diambil dari bounding box kandidat.
====================================================================
"""

import cv2
import numpy as np
from . import config
from . import money_validator


def _buat_mask_tepi(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Membuat citra tepi dan hasil morfologi dari frame.
    Fungsi ini dipakai oleh deteksi ROI dan debug window.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE membantu tepi uang tetap muncul saat cahaya tidak rata/redup.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    # Canny adaptif dari median intensitas.
    median_v = np.median(blur)
    sigma = 0.33
    ambang_bawah = int(max(20, (1.0 - sigma) * median_v))
    ambang_atas = int(min(180, (1.0 + sigma) * median_v))
    canny = cv2.Canny(blur, ambang_bawah, ambang_atas)

    # Kernel jangan terlalu besar. Kernel 9x9 sering menggabungkan uang
    # dengan tangan/meja sehingga bentuk uang hilang.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilasi = cv2.dilate(canny, kernel, iterations=1)
    closing = cv2.morphologyEx(dilasi, cv2.MORPH_CLOSE, kernel, iterations=2)

    return canny, closing


def cari_kotak_uang(frame: np.ndarray):
    """
    Mendeteksi ROI uang dari frame kamera.

    Return:
        (roi, bbox) jika kandidat uang ditemukan
        None jika tidak ada bentuk yang cukup mirip uang

    Catatan:
    Validasi dibuat lebih longgar karena uang bisa miring, terlipat,
    sebagian tertutup jari, atau tepinya kurang jelas.
    """
    if frame is None or frame.size == 0:
        return None

    canny, closing = _buat_mask_tepi(frame)

    contours, _ = cv2.findContours(
        closing,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    frame_area = frame.shape[0] * frame.shape[1]
    kandidat = []

    for cnt in contours:
        contour_area = cv2.contourArea(cnt)
        if contour_area < config.LUAS_ROI_MIN:
            continue

        # minAreaRect lebih cocok untuk objek miring dibanding boundingRect biasa.
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect

        if rw <= 0 or rh <= 0:
            continue

        rect_area = rw * rh
        if rect_area <= 0:
            continue

        if rect_area > frame_area * 0.85:
            continue

        aspect_ratio = max(rw, rh) / min(rw, rh)
        if not (config.ASPEK_RASIO_MIN <= aspect_ratio <= config.ASPEK_RASIO_MAX):
            continue

        rectangularity = contour_area / rect_area

        # Dibuat longgar karena uang asli punya gambar/tepi terputus,
        # dan kontur hasil Canny jarang mengisi kotak 100%.
        if rectangularity < money_validator.RECTANGULARITY_MIN:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Hindari ROI terlalu kecil setelah boundingRect.
        if w * h < config.LUAS_ROI_MIN:
            continue

        # Skor kandidat: gabungan area dan kemiripan bentuk uang.
        # Semakin besar area dan rectangularity, semakin diprioritaskan.
        skor = (contour_area * 0.7) + (rect_area * rectangularity * 0.3)

        kandidat.append((skor, x, y, w, h))

    if not kandidat:
        return None

    kandidat.sort(key=lambda item: item[0], reverse=True)
    _, x, y, w, h = kandidat[0]

    pad = 12
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(frame.shape[1], x + w + pad)
    y2 = min(frame.shape[0], y + h + pad)

    roi = frame[y1:y2, x1:x2]
    bbox = (x1, y1, x2 - x1, y2 - y1)

    if roi is None or roi.size == 0:
        return None

    return roi, bbox


def ambil_citra_debug(frame: np.ndarray) -> tuple:
    """
    Menghasilkan citra hasil Canny dan Morfologi untuk jendela debug.
    """
    canny, closing = _buat_mask_tepi(frame)

    canny_vis = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    morph_vis = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)

    return canny_vis, morph_vis


def ambil_roi_guide(frame: np.ndarray):
    """
    Mengambil ROI tetap dari kotak panduan tengah frame.

    Mode ini dibuat sebagai mode utama untuk demo/asistensi karena ROI
    otomatis berbasis kontur sangat sensitif terhadap cahaya, background,
    kemiringan uang, dan tangan pengguna. Dengan guide, pengguna cukup
    meletakkan uang di dalam kotak.

    Return:
        roi  : area gambar di dalam kotak panduan
        bbox : koordinat kotak (x, y, w, h)
    """
    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]

    x1 = int(w * config.GUIDE_X_RATIO)
    y1 = int(h * config.GUIDE_Y_RATIO)
    gw = int(w * config.GUIDE_W_RATIO)
    gh = int(h * config.GUIDE_H_RATIO)

    x2 = min(w, x1 + gw)
    y2 = min(h, y1 + gh)

    roi = frame[y1:y2, x1:x2]
    bbox = (x1, y1, x2 - x1, y2 - y1)

    if roi is None or roi.size == 0:
        return None

    return roi, bbox


def inisialisasi_tracker():
    """
    Membuat objek KCF Tracker secara dinamis untuk menghindari kegagalan lintas versi OpenCV.
    
    Return:
        Objek tracker KCF jika tersedia, atau None jika modul dinonaktifkan di sistem.
    """
    try:
        return cv2.TrackerKCF.create()
    except AttributeError:
        try:
            return cv2.legacy.TrackerKCF_create()
        except AttributeError:
            return None