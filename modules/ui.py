"""
====================================================================
ui.py — Antarmuka Visual OpenCV (HUD, Trackbar, Debug Mode)
====================================================================
File ini mengatur semua tampilan visual:
- Jendela utama OpenCV
- Trackbar kalibrasi HSV
- Bounding box ROI
- HUD hasil deteksi
- Info hybrid HSV + Template + Final
- Debug Canny/Morfologi
====================================================================
"""

import cv2
import numpy as np

from . import config
from . import hybrid_decision


# ------------------------------------------------------------------ #
#  NAMA JENDELA OPENCV                                               #
# ------------------------------------------------------------------ #

NAMA_JENDELA_UTAMA  = "Vision-to-Audio Bridge | Pengenalan Rupiah"
NAMA_JENDELA_DEBUG  = "Debug: Canny | Morfologi"
NAMA_JENDELA_KALIBR = "Kalibrasi HSV"


# ------------------------------------------------------------------ #
#  LABEL STATUS                                                       #
# ------------------------------------------------------------------ #

LABEL_TIDAK_VALID = {
    "Tidak terdeteksi",
    "Tidak yakin",
    "Objek bukan uang",
    "Cahaya Kurang",
    "Cahaya kurang",
    "Arahkan uang ke kamera",
    "Letakkan uang di dalam kotak",
    "Tidak ada template",
    "Tidak yakin (template)",
    None,
}


def is_nominal_valid(label: str) -> bool:
    """
    Mengecek apakah label adalah nominal uang atau status gagal.
    """
    return label not in LABEL_TIDAK_VALID


# ------------------------------------------------------------------ #
#  TRACKBAR KALIBRASI HSV                                            #
# ------------------------------------------------------------------ #

def buat_trackbar() -> None:
    """
    Membuat jendela kalibrasi HSV dengan trackbar interaktif.
    """
    cv2.namedWindow(NAMA_JENDELA_KALIBR, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(NAMA_JENDELA_KALIBR, 420, 300)

    cv2.createTrackbar("H Min", NAMA_JENDELA_KALIBR,   0, 179, lambda x: None)
    cv2.createTrackbar("S Min", NAMA_JENDELA_KALIBR,  40, 255, lambda x: None)
    cv2.createTrackbar("V Min", NAMA_JENDELA_KALIBR,  40, 255, lambda x: None)

    cv2.createTrackbar("H Max", NAMA_JENDELA_KALIBR, 179, 179, lambda x: None)
    cv2.createTrackbar("S Max", NAMA_JENDELA_KALIBR, 255, 255, lambda x: None)
    cv2.createTrackbar("V Max", NAMA_JENDELA_KALIBR, 255, 255, lambda x: None)


def baca_trackbar() -> tuple:
    """
    Membaca nilai semua trackbar HSV.
    Return:
        lower, upper dalam format numpy array uint8.
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
#  BOUNDING BOX ROI                                                   #
# ------------------------------------------------------------------ #

def gambar_bounding_box(
    frame: np.ndarray,
    bbox: tuple,
    label: str,
    is_fallback: bool
) -> None:
    """
    Menggambar bounding box di sekitar ROI. (Text dihapus sesuai request)
    """
    if bbox is None:
        return

    x, y, w, h = bbox

    if is_nominal_valid(label):
        warna_box = (0, 220, 50)      # Hijau
    elif is_fallback:
        warna_box = (0, 140, 255)     # Oranye
    else:
        warna_box = (0, 220, 220)     # Kuning

    cv2.rectangle(frame, (x, y), (x + w, y + h), warna_box, 2)


# ------------------------------------------------------------------ #
#  HUD HASIL DETEKSI                                                  #
# ------------------------------------------------------------------ #

def gambar_hud(frame: np.ndarray, label: str, fps: float) -> None:
    """
    (Dikosongkan sesuai request agar teks tidak menghalangi video di frontend)
    """
    pass


# ------------------------------------------------------------------ #
#  INFO HYBRID HSV + TEMPLATE + FINAL                                 #
# ------------------------------------------------------------------ #

def gambar_info_hybrid(
    frame: np.ndarray,
    hasil_hsv: str,
    hasil_template: str,
    skor_template: float,
    sumber: str,
    label_final: str = None
) -> None:
    """
    (Dikosongkan sesuai request agar teks tidak menghalangi video di frontend)
    """
    pass


# ------------------------------------------------------------------ #
#  INFO DEBUG PIXEL / SKOR WARNA                                      #
# ------------------------------------------------------------------ #

def gambar_info_debug_pixel(frame: np.ndarray, pixel_counts: list) -> None:
    """
    Menampilkan daftar persentase/skor warna di pojok kiri atas.

    Format item bisa:
    - (nama, persen, skor)
    - (nama, pixel)
    """
    y_awal = 20
    jarak = 18

    for i, item in enumerate(pixel_counts):
        if len(item) == 3:
            nama, persen, skor = item
            teks = f"{nama}: {persen:.1f}% skor={skor:.1f}"
        elif len(item) == 2:
            nama, pixel = item
            teks = f"{nama}: {pixel} px"
        else:
            continue

        y_pos = y_awal + i * jarak

        cv2.putText(
            frame,
            teks,
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )


# ------------------------------------------------------------------ #
#  PREVIEW ROI                                                        #
# ------------------------------------------------------------------ #

def tampilkan_preview_roi(roi_asli: np.ndarray, roi_processed: np.ndarray = None) -> None:
    """
    Menampilkan preview ROI asli dan ROI preprocessing.
    Aman dipanggil hanya saat debug/preprocessing ingin dicek.
    """
    if roi_asli is not None and roi_asli.size > 0:
        cv2.imshow("ROI Asli", roi_asli)

    if roi_processed is not None and roi_processed.size > 0:
        cv2.imshow("ROI Preprocessing", roi_processed)


def tutup_preview_roi() -> None:
    """
    Menutup jendela preview ROI.
    """
    for nama in ["ROI Asli", "ROI Preprocessing"]:
        try:
            cv2.destroyWindow(nama)
        except Exception:
            pass


# ------------------------------------------------------------------ #
#  JENDELA DEBUG CANNY / MORFOLOGI                                    #
# ------------------------------------------------------------------ #

def tampilkan_debug_window(canny_vis: np.ndarray, morph_vis: np.ndarray) -> None:
    """
    Menampilkan jendela debug Canny dan Morfologi berdampingan.
    """
    if canny_vis is None or morph_vis is None:
        return

    if canny_vis.shape[0] != morph_vis.shape[0]:
        morph_vis = cv2.resize(
            morph_vis,
            (morph_vis.shape[1], canny_vis.shape[0])
        )

    canny_labeled = canny_vis.copy()
    morph_labeled = morph_vis.copy()

    cv2.putText(
        canny_labeled,
        "CANNY EDGE",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        morph_labeled,
        "MORFOLOGI",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 200, 255),
        2
    )

    panel_debug = np.hstack([canny_labeled, morph_labeled])
    cv2.imshow(NAMA_JENDELA_DEBUG, panel_debug)


def tutup_debug_window() -> None:
    """
    Menutup jendela debug jika sedang terbuka.
    """
    try:
        cv2.destroyWindow(NAMA_JENDELA_DEBUG)
    except Exception:
        pass