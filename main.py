"""
====================================================================
main.py — Orchestrator Utama Vision-to-Audio Bridge
====================================================================
Versi v5 HYBRID (Guide + HSV + Template Matching):

Pipeline deteksi per frame:
  1. Baca frame kamera
  2. Resize + Median Blur
  3. Ambil ROI dari kotak guide (mode GUIDE) atau kontur (mode AUTO)
  4. Jika ROI tidak ada → tampilkan "Letakkan uang di dalam kotak"
  5. Cek pencahayaan (channel V pada HSV)
  6. Jalankan deteksi warna HSV → hasil_hsv
  7. Jalankan template matching → hasil_template, skor_template
  8. Gabungkan keduanya via hybrid_decision → label_final
  9. Tampilkan info hybrid di layar (HSV / Template / Sumber)
  10. TTS membacakan label_final
  11. Tampilkan frame ke layar

Tombol keyboard:
  Q = keluar
  D = debug Canny/Morfologi
  T = kalibrasi trackbar HSV
  M = ganti mode ROI: GUIDE / AUTO
  H = toggle tampilan info hybrid di layar
====================================================================
"""

import cv2
import time
import numpy as np

from modules import config
from modules import roi_detector
from modules import color_detector
from modules import lighting
from modules import speech
from modules import ui
from modules import money_validator
from modules import template_matcher
from modules import hybrid_decision
from modules.preprocessing import preprocess_roi

# Label-label yang bukan nominal uang (tidak dibacakan TTS)
STATUS_NON_NOMINAL = {
    "Letakkan uang di dalam kotak",
    "Arahkan uang ke kamera",
    "Cahaya Kurang",
    "Objek bukan uang",
    "Tidak terdeteksi",
    "Tidak yakin",
}


def buka_kamera():
    """
    Membuka kamera dengan beberapa backend/index.
    Di Windows, CAP_DSHOW sering lebih stabil daripada default MSMF.
    """
    kandidat_backend = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    kandidat_index   = [0, 1, 2]

    for backend in kandidat_backend:
        for index in kandidat_index:
            kamera = cv2.VideoCapture(index, backend)
            if kamera.isOpened():
                kamera.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
                kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
                kamera.set(cv2.CAP_PROP_FPS, 30)

                ret, _ = kamera.read()
                if ret:
                    print(f"[INFO] Kamera aktif: index={index}, backend={backend}")
                    return kamera

            kamera.release()

    return None


def proses_roi_hybrid(roi: np.ndarray, metode_roi: str) -> tuple:
    """
    Memproses ROI melalui pipeline hybrid:
    ROI asli → preprocessing → cek cahaya → HSV → Template → Hybrid Decision.

    Mengembalikan:
        tuple (label_final, hasil_hsv, hasil_template, skor_template, sumber)
    """

    # Nilai default jika ROI kosong
    if roi is None or roi.size == 0:
        return (
            "Arahkan uang ke kamera",
            "Arahkan uang ke kamera",
            "Tidak ada template",
            0.0,
            "Tidak yakin"
        )

    # ------------------------------------------------------------
    # LANGKAH 1: PREPROCESSING ROI
    # ------------------------------------------------------------
    # Ini bagian penting yang sebelumnya belum dipakai.
    # Tujuannya agar citra lebih stabil terhadap cahaya.
    roi_processed = preprocess_roi(roi)

    # ------------------------------------------------------------
    # LANGKAH 2: CEK PENCAHAYAAN
    # ------------------------------------------------------------
    # Cek pencahayaan dilakukan setelah preprocessing agar gambar
    # yang awalnya agak redup masih bisa diperbaiki terlebih dahulu.
    roi_hsv = cv2.cvtColor(roi_processed, cv2.COLOR_BGR2HSV)

    if lighting.cek_kondisi_cahaya(roi_hsv):
        return (
            "Cahaya Kurang",
            "Cahaya Kurang",
            "Tidak ada template",
            0.0,
            "Tidak yakin"
        )

    # ------------------------------------------------------------
    # LANGKAH 3: VALIDASI OBJEK
    # ------------------------------------------------------------
    # Pada mode AUTO validasi tetap perlu karena ROI bisa berasal
    # dari objek apa pun.
    #
    # Pada mode GUIDE, validasi hanya dijalankan jika
    # config.VALIDASI_KETAT_GUIDE = True.
    #
    # Validasi menggunakan ROI hasil preprocessing agar tekstur
    # dan variasi warnanya lebih terbaca.
    if metode_roi == "AUTO":
        tekstur_valid = money_validator.validasi_tekstur_uang(roi_processed)
        warna_valid = money_validator.validasi_variasi_warna(roi_processed)

        if not tekstur_valid or not warna_valid:
            return (
                "Objek bukan uang",
                "Objek bukan uang",
                "Tidak ada template",
                0.0,
                "Tidak yakin"
            )

    elif metode_roi == "GUIDE" and config.VALIDASI_KETAT_GUIDE:
        tekstur_valid = money_validator.validasi_tekstur_uang(roi_processed)
        warna_valid = money_validator.validasi_variasi_warna(roi_processed)

        if not tekstur_valid or not warna_valid:
            return (
                "Objek bukan uang",
                "Objek bukan uang",
                "Tidak ada template",
                0.0,
                "Tidak yakin"
            )

    # ------------------------------------------------------------
    # LANGKAH 4: DETEKSI HSV
    # ------------------------------------------------------------
    # Gunakan ROI hasil preprocessing, bukan ROI asli.
    hasil_hsv = color_detector.tentukan_nominal(roi_processed)

    # ------------------------------------------------------------
    # LANGKAH 5: TEMPLATE MATCHING
    # ------------------------------------------------------------
    # Gunakan ROI hasil preprocessing agar template matching
    # lebih tahan terhadap perubahan cahaya.
    hasil_template, skor_template, _ = template_matcher.cocokkan_template(roi_processed)

    # ------------------------------------------------------------
    # LANGKAH 6: HYBRID DECISION
    # ------------------------------------------------------------
    label_final, sumber = hybrid_decision.gabungkan_keputusan(
        hasil_hsv,
        hasil_template,
        skor_template
    )

    return label_final, hasil_hsv, hasil_template, skor_template, sumber


def main() -> None:
    """
    Loop utama program Vision-to-Audio Bridge.
    """
    kamera = buka_kamera()

    if kamera is None:
        print("[ERROR] Kamera tidak dapat dibuka.")
        print("[SARAN] Tutup Camera/Zoom/Meet/OBS, lalu jalankan ulang.")
        print("[SARAN] Cek izin kamera: Settings > Privacy > Camera.")
        return

    print("[INFO] Kamera berhasil dibuka.")
    print("[INFO] Tombol: Q=Keluar | D=Debug | T=Kalibrasi | M=Mode | H=Info Hybrid")
    print("[INFO] Mode awal: GUIDE. Letakkan uang asli di dalam kotak panduan.")

    # ---- State Program ---- #
    mode_debug         = False
    mode_kalibrasi     = False
    mode_roi           = "GUIDE"       # GUIDE = kotak panduan manual
    tampilkan_hybrid   = config.TAMPILKAN_INFO_HYBRID  # H untuk toggle

    # ---- Variabel FPS ---- #
    waktu_fps_sebelumnya = time.time()
    fps_aktif            = 0.0

    # ---- Loop Utama ---- #
    while True:

        # ---- Langkah 1: Baca Frame ---- #
        ret, frame = kamera.read()
        if not ret:
            print("[WARNING] Frame gagal diambil, mencoba lagi...")
            time.sleep(0.3)
            continue

        frame      = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        frame_blur = cv2.medianBlur(frame, 5)

        roi         = None
        bbox        = None
        is_fallback = (mode_roi == "GUIDE")

        # ---- Langkah 2: Ambil ROI ---- #
        if mode_roi == "GUIDE":
            # Mode GUIDE: ROI diambil dari kotak panduan tetap di tengah frame
            hasil_roi  = roi_detector.ambil_roi_guide(frame_blur)
            label_awal = "Letakkan uang di dalam kotak"
        else:
            # Mode AUTO: ROI dicari otomatis dari kontur Canny
            hasil_roi  = roi_detector.cari_kotak_uang(frame_blur)
            label_awal = "Arahkan uang ke kamera"

        if hasil_roi is not None:
            roi, bbox = hasil_roi

        # ---- Langkah 3: Proses atau Tampilkan Status Kosong ---- #
        if roi is None or bbox is None:
            # ROI tidak tersedia → tampilkan pesan panduan
            label_final    = label_awal
            hasil_hsv      = label_awal
            hasil_template = "Tidak ada template"
            skor_template  = 0.0
            sumber         = "Tidak yakin"

            # Tetap tampilkan kotak guide agar user tahu posisi yang benar
            hasil_guide = roi_detector.ambil_roi_guide(frame_blur)
            if hasil_guide is not None:
                _, bbox = hasil_guide
                is_fallback = True
            else:
                bbox = (0, 0, config.FRAME_WIDTH, config.FRAME_HEIGHT)

        else:
            # ---- Langkah 4–8: Pipeline Hybrid ---- #
            label_final, hasil_hsv, hasil_template, skor_template, sumber = \
                proses_roi_hybrid(roi, mode_roi)

        # ---- Langkah 9: Render Tampilan ---- #

        # Gambar kotak bounding box
        ui.gambar_bounding_box(frame, bbox, label_final, is_fallback=is_fallback)

        # Gambar HUD bawah dengan label final
        ui.gambar_hud(frame, label_final, fps_aktif)

        # Label mode ROI di atas layar
        cv2.putText(
            frame,
            f"MODE ROI: {mode_roi}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (0, 255, 255) if mode_roi == "GUIDE" else (0, 220, 50),
            2, cv2.LINE_AA
        )

        # Info hybrid (HSV / Template / Sumber) jika toggle aktif
        if tampilkan_hybrid:
            ui.gambar_info_hybrid(
                frame,
                hasil_hsv,
                hasil_template,
                skor_template,
                sumber
            )

        # Preview mask HSV saat kalibrasi aktif
        if mode_kalibrasi and roi is not None and roi.size > 0:
            lower_kalibrasi, upper_kalibrasi = ui.baca_trackbar()
            roi_hsv_kal    = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask_kalibrasi = cv2.inRange(roi_hsv_kal, lower_kalibrasi, upper_kalibrasi)
            mask_kecil     = cv2.resize(mask_kalibrasi, (160, 120))
            mask_bgr       = cv2.cvtColor(mask_kecil, cv2.COLOR_GRAY2BGR)
            frame[0:120, config.FRAME_WIDTH - 160:config.FRAME_WIDTH] = mask_bgr

        # Debug window Canny + Morfologi
        if mode_debug:
            canny_vis, morph_vis = roi_detector.ambil_citra_debug(frame_blur)
            ui.tampilkan_debug_window(canny_vis, morph_vis)

        # ---- Hitung FPS ---- #
        waktu_sekarang       = time.time()
        selang_waktu         = waktu_sekarang - waktu_fps_sebelumnya
        if selang_waktu > 0:
            fps_aktif = 1.0 / selang_waktu
        waktu_fps_sebelumnya = waktu_sekarang

        # ---- Langkah 10: TTS — hanya bicara jika label adalah nominal valid ---- #
        # Status seperti "Cahaya Kurang" dan "Objek bukan uang" tidak dibacakan
        if label_final not in STATUS_NON_NOMINAL:
            speech.bicara_nominal(label_final)

        # ---- Langkah 11: Tampilkan Frame ---- #
        cv2.imshow(ui.NAMA_JENDELA_UTAMA, frame)

        # ---- Tombol Keyboard ---- #
        tombol = cv2.waitKey(1) & 0xFF

        if tombol == ord("q"):
            print("[INFO] Keluar dari program.")
            break

        elif tombol == ord("d"):
            mode_debug = not mode_debug
            print(f"[INFO] Mode Debug: {'AKTIF' if mode_debug else 'NONAKTIF'}")
            if not mode_debug:
                ui.tutup_debug_window()

        elif tombol == ord("t"):
            mode_kalibrasi = not mode_kalibrasi
            print(f"[INFO] Kalibrasi HSV: {'AKTIF' if mode_kalibrasi else 'NONAKTIF'}")
            if mode_kalibrasi:
                ui.buat_trackbar()
            else:
                try:
                    cv2.destroyWindow(ui.NAMA_JENDELA_KALIBR)
                except Exception:
                    pass

        elif tombol == ord("m"):
            # Toggle antara mode GUIDE dan AUTO
            mode_roi = "AUTO" if mode_roi == "GUIDE" else "GUIDE"
            print(f"[INFO] Mode ROI: {mode_roi}")

        elif tombol == ord("h"):
            # Toggle tampilan info hybrid di layar
            tampilkan_hybrid = not tampilkan_hybrid
            print(f"[INFO] Info Hybrid: {'TAMPIL' if tampilkan_hybrid else 'SEMBUNYI'}")

    # ---- Bersihkan Resource ---- #
    kamera.release()
    cv2.destroyAllWindows()
    print("[INFO] Kamera dibebaskan. Sampai jumpa!")


if __name__ == "__main__":
    main()
