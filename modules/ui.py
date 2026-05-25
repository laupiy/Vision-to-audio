"""
====================================================================
ui.py — Antarmuka Visual OpenCV (HUD, Trackbar, Debug Mode)
====================================================================
Versi v5 tambahan:
  - gambar_info_hybrid() : menampilkan baris HSV / Template / Sumber
    di bagian atas layar untuk transparansi keputusan sistem.
====================================================================
"""

import cv2
import numpy as np
from . import config


# ------------------------------------------------------------------ #
#  NAMA JENDELA OPENCV                                               #
# ------------------------------------------------------------------ #

NAMA_JENDELA_UTAMA  = "Vision-to-Audio Bridge | Pengenalan Rupiah"
NAMA_JENDELA_DEBUG  = "Debug: Canny | Morfologi"
NAMA_JENDELA_KALIBR = "Kalibrasi HSV"


# ------------------------------------------------------------------ #
#  TRACKBAR KALIBRASI HSV                                            #
# ------------------------------------------------------------------ #

def buat_trackbar() -> None:
    """
    Membuat jendela kalibrasi HSV dengan trackbar interaktif.
    """
    cv2.namedWindow(NAMA_JENDELA_KALIBR, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(NAMA_JENDELA_KALIBR, 400, 300)

    cv2.createTrackbar("H Min", NAMA_JENDELA_KALIBR,   0, 179, lambda x: None)
    cv2.createTrackbar("S Min", NAMA_JENDELA_KALIBR,  80, 255, lambda x: None)
    cv2.createTrackbar("V Min", NAMA_JENDELA_KALIBR,  80, 255, lambda x: None)
    cv2.createTrackbar("H Max", NAMA_JENDELA_KALIBR,  10, 179, lambda x: None)
    cv2.createTrackbar("S Max", NAMA_JENDELA_KALIBR, 255, 255, lambda x: None)
    cv2.createTrackbar("V Max", NAMA_JENDELA_KALIBR, 255, 255, lambda x: None)


def baca_trackbar() -> tuple:
    """
    Membaca nilai saat ini dari semua trackbar kalibrasi HSV.
    """
    h_min = cv2.getTrackbarPos("H Min", NAMA_JENDELA_KALIBR)
    s_min = cv2.getTrackbarPos("S Min", NAMA_JENDELA_KALIBR)
    v_min = cv2.getTrackbarPos("V Min", NAMA_JENDELA_KALIBR)
    h_max = cv2.getTrackbarPos("H Max", NAMA_JENDELA_KALIBR)
    s_max = cv2.getTrackbarPos("S Max", NAMA_JENDELA_KALIBR)
    v_max = cv2.getTrackbarPos("V Max", NAMA_JENDELA_KALIBR)

    lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper = np.array([h_max, s_max, v_max], dtype=np.uint8)

    return lower, upper


# ------------------------------------------------------------------ #
#  BOUNDING BOX DINAMIS                                              #
# ------------------------------------------------------------------ #

def gambar_bounding_box(
    frame      : np.ndarray,
    bbox       : tuple,
    label      : str,
    is_fallback: bool
) -> None:
    """
    Menggambar bounding box di sekitar ROI dengan warna sesuai status.

    Warna kotak:
      - Hijau  : Nominal teridentifikasi
      - Oranye : Mode guide (fallback/manual)
      - Kuning : Kontur ada tapi label ambigu
    """
    x, y, w, h = bbox

    label_tidak_valid = {
        "Tidak terdeteksi", "Tidak yakin", "Cahaya Kurang",
        "Objek bukan uang", "Arahkan uang ke kamera",
        "Letakkan uang di dalam kotak",
    }

    if label not in label_tidak_valid:
        warna_box = (0, 220, 50)     # Hijau → nominal terdeteksi
    elif is_fallback:
        warna_box = (0, 140, 255)    # Oranye → mode guide
    else:
        warna_box = (0, 220, 220)    # Kuning → tidak cocok

    cv2.rectangle(frame, (x, y), (x + w, y + h), warna_box, 2)

    teks_kecil = "ROI: GUIDE" if is_fallback else "ROI: KONTUR"
    cv2.putText(
        frame, teks_kecil,
        (x, max(0, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4, warna_box, 1, cv2.LINE_AA
    )


# ------------------------------------------------------------------ #
#  PANEL HUD BAWAH (HEAD-UP DISPLAY)                                 #
# ------------------------------------------------------------------ #

def gambar_hud(frame: np.ndarray, label: str, fps: float) -> None:
    """
    Menampilkan panel HUD semi-transparan di bagian bawah frame.
    Berisi label nominal besar, FPS, dan instruksi tombol keyboard.
    """
    tinggi_panel = 80
    y_panel      = config.FRAME_HEIGHT - tinggi_panel

    # Overlay gelap semi-transparan
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_panel),
                  (config.FRAME_WIDTH, config.FRAME_HEIGHT), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Warna teks sesuai status
    label_tidak_valid = {
        "Tidak terdeteksi", "Tidak yakin", "Cahaya Kurang",
        "Objek bukan uang", "Arahkan uang ke kamera",
        "Letakkan uang di dalam kotak",
    }
    if label == "Cahaya Kurang":
        warna_teks = (0, 100, 255)    # Oranye
    elif label in label_tidak_valid:
        warna_teks = (100, 100, 100)  # Abu-abu
    else:
        warna_teks = (0, 255, 100)    # Hijau terang

    # Teks nominal besar di tengah HUD
    font       = cv2.FONT_HERSHEY_DUPLEX
    skala_font = 1.2
    tebal_font = 2
    (lebar_teks, tinggi_teks), _ = cv2.getTextSize(label, font, skala_font, tebal_font)
    x_teks = (config.FRAME_WIDTH - lebar_teks) // 2
    y_teks = y_panel + tinggi_panel // 2 + tinggi_teks // 2

    cv2.putText(frame, label, (x_teks, y_teks), font, skala_font,
                warna_teks, tebal_font, cv2.LINE_AA)

    # FPS pojok kanan bawah
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (config.FRAME_WIDTH - 100, config.FRAME_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Instruksi keyboard pojok kiri bawah
    cv2.putText(frame, "Q:Keluar | D:Debug | T:Kalibrasi | M:Mode | H:Info Hybrid",
                (10, config.FRAME_HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)


# ------------------------------------------------------------------ #
#  INFO HYBRID: HSV vs TEMPLATE (BARU)                               #
# ------------------------------------------------------------------ #

def gambar_info_hybrid(
    frame         : np.ndarray,
    hasil_hsv     : str,
    hasil_template: str,
    skor_template : float,
    sumber        : str
) -> None:
    """
    Menampilkan baris informasi perbandingan HSV vs Template di layar.

    Ditampilkan di bagian atas frame agar tidak tumpang tindih dengan HUD.
    Berguna saat presentasi/demo ke dosen untuk menjelaskan cara kerja
    sistem hybrid (metode gabungan).

    Baris yang ditampilkan:
      HSV     : Rp20.000
      Template: Rp20.000 [0.72]
      Sumber  : Template

    Parameter:
        frame          : numpy array BGR
        hasil_hsv      : label dari color_detector
        hasil_template : label dari template_matcher
        skor_template  : skor float 0.0–1.0
        sumber         : "HSV", "Template", atau "Tidak yakin"
    """
    # Latar gelap semi-transparan di bagian atas frame untuk info hybrid
    tinggi_overlay = 62   # Cukup untuk 3 baris teks kecil
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 30), (320, 30 + tinggi_overlay),
                  (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Tentukan warna skor template: hijau jika kuat, abu-abu jika lemah
    from . import hybrid_decision
    warna_template = (
        (100, 255, 100)   # Hijau muda → skor kuat
        if skor_template >= hybrid_decision.THRESHOLD_TEMPLATE_SEDANG
        else (150, 150, 150)  # Abu-abu → skor lemah
    )

    # Baris 1: Hasil HSV
    cv2.putText(
        frame,
        f"HSV     : {hasil_hsv}",
        (8, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (0, 255, 255), 1, cv2.LINE_AA   # Kuning
    )

    # Baris 2: Hasil Template + Skor
    cv2.putText(
        frame,
        f"Template: {hasil_template}  [{skor_template:.2f}]",
        (8, 68),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        warna_template, 1, cv2.LINE_AA
    )

    # Baris 3: Sumber keputusan
    cv2.putText(
        frame,
        f"Sumber  : {sumber}",
        (8, 86),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (200, 200, 200), 1, cv2.LINE_AA  # Abu-abu terang
    )


# ------------------------------------------------------------------ #
#  INFO DEBUG PIXEL COUNT                                            #
# ------------------------------------------------------------------ #

def gambar_info_debug_pixel(frame: np.ndarray, pixel_counts: list) -> None:
    """
    Menampilkan jumlah piksel cocok setiap nominal di pojok kiri atas.
    """
    y_awal = 20
    jarak  = 18

    for i, (nama, persen, skor) in enumerate(pixel_counts):
        teks  = f"{nama}: {persen:.1f}% (skor {skor:.1f})"
        y_pos = y_awal + i * jarak
        cv2.putText(frame, teks, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)


# ------------------------------------------------------------------ #
#  JENDELA DEBUG MODE                                                #
# ------------------------------------------------------------------ #

def tampilkan_debug_window(canny_vis: np.ndarray, morph_vis: np.ndarray) -> None:
    """
    Menampilkan jendela sekunder Canny | Morfologi berdampingan.
    """
    if canny_vis.shape[0] != morph_vis.shape[0]:
        morph_vis = cv2.resize(morph_vis, (morph_vis.shape[1], canny_vis.shape[0]))

    canny_labeled = canny_vis.copy()
    morph_labeled = morph_vis.copy()

    cv2.putText(canny_labeled, "CANNY EDGE",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(morph_labeled, "MORFOLOGI",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    panel_debug = np.hstack([canny_labeled, morph_labeled])
    cv2.imshow(NAMA_JENDELA_DEBUG, panel_debug)


def tutup_debug_window() -> None:
    """Menutup jendela debug jika sedang terbuka."""
    try:
        cv2.destroyWindow(NAMA_JENDELA_DEBUG)
    except Exception:
        pass
