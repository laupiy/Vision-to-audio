"""
====================================================================
template_matcher.py — Pencocokan Template Angka Nominal Uang
====================================================================
Modul ini mendeteksi nominal uang menggunakan teknik Template Matching
dari OpenCV (cv2.matchTemplate), bukan analisis warna HSV.

Cara kerja template matching:
  1. Muat gambar template angka dari folder templates/.
     Contoh: templates/100000.png, templates/50000.png, dst.
  2. Ubah ROI kamera dan template ke grayscale agar tidak terpengaruh
     perbedaan warna akibat cahaya yang berubah-ubah.
  3. Coba beberapa ukuran template (multi-scale) agar cocok meskipun
     uang di kamera lebih dekat atau jauh dari kamera.
  4. Gunakan cv2.TM_CCOEFF_NORMED:
     - Hasilnya range 0.0 – 1.0
     - Semakin mendekati 1.0 = semakin mirip template
  5. Kembalikan nominal dengan skor tertinggi dan semua skor untuk
     ditampilkan di layar.

Mengapa tidak bergantung pada warna?
  → Template matching mencari pola angka/gambar, bukan warna.
  → Saat uang Rp20.000 terlihat pucat/abu-abu karena cahaya redup,
    HSV gagal mengenali hijau → salah baca Rp2.000.
  → Template matching tetap bisa mengenali ANGKA "20000" di uang.

Jika folder templates/ kosong atau file tidak ditemukan:
  → Sistem tetap berjalan tanpa error.
  → Fungsi mengembalikan "Tidak ada template".
====================================================================
"""

import os
import cv2
import numpy as np


# ------------------------------------------------------------------ #
#  KONFIGURASI                                                        #
# ------------------------------------------------------------------ #

# Lokasi folder template relatif dari file ini
# Folder templates/ harus berada di root project (sejajar main.py)
FOLDER_TEMPLATE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "templates"
)

# Daftar nama file template dan label nominalnya
# Key   = nama file tanpa ekstensi (cocok dengan nama file .png)
# Value = label nominal yang ditampilkan di layar
DAFTAR_TEMPLATE = {
    "100000": "Rp100.000",
    "50000" : "Rp50.000",
    "20000" : "Rp20.000",
    "10000" : "Rp10.000",
    "5000"  : "Rp5.000",
    "2000"  : "Rp2.000",
    "1000"  : "Rp1.000",
}

# Skala yang dicoba saat multi-scale matching.
# Nilai < 1.0 = memperkecil template (uang jauh/kecil di kamera).
# Nilai > 1.0 = memperbesar template (uang dekat/besar di kamera).
SKALA_MULTI = [0.5, 0.65, 0.8, 1.0, 1.2, 1.5]

# Nilai minimum skor agar hasil template matching dianggap valid.
# Range skor: 0.0 – 1.0. Di bawah nilai ini dianggap tidak cocok.
SKOR_MINIMUM = 0.35


# ------------------------------------------------------------------ #
#  MUAT TEMPLATE SAAT MODUL DIIMPOR                                  #
# ------------------------------------------------------------------ #

def _muat_semua_template() -> dict:
    """
    Memuat semua file template gambar dari folder templates/.

    Setiap template dimuat sekali saat program pertama kali dijalankan
    (bukan setiap frame) untuk menghemat waktu pemrosesan.

    Mengembalikan:
        dict { nama_file: (label_nominal, gambar_grayscale) }
        Dict kosong jika folder tidak ada atau tidak ada file valid.
    """
    hasil = {}

    # Jika folder templates/ tidak ada, buat dulu agar tidak error
    if not os.path.isdir(FOLDER_TEMPLATE):
        print(f"[INFO] Folder templates/ tidak ditemukan di: {FOLDER_TEMPLATE}")
        print("[INFO] Buat folder templates/ dan isi dengan gambar nominal.")
        return hasil

    for kode, label in DAFTAR_TEMPLATE.items():
        path_png = os.path.join(FOLDER_TEMPLATE, f"{kode}.png")
        path_jpg = os.path.join(FOLDER_TEMPLATE, f"{kode}.jpg")

        # Coba format PNG dulu, lalu JPG
        path_file = None
        if os.path.isfile(path_png):
            path_file = path_png
        elif os.path.isfile(path_jpg):
            path_file = path_jpg

        if path_file is None:
            # File tidak ditemukan → lewati, tidak error
            continue

        # Baca gambar dalam grayscale langsung (lebih efisien)
        gambar = cv2.imread(path_file, cv2.IMREAD_GRAYSCALE)

        if gambar is None:
            print(f"[WARNING] Gagal membaca template: {path_file}")
            continue

        hasil[kode] = (label, gambar)
        print(f"[INFO] Template dimuat: {label} ({gambar.shape[1]}x{gambar.shape[0]}px)")

    if not hasil:
        print("[INFO] Tidak ada template yang berhasil dimuat.")
        print("[INFO] Template matching tidak akan berjalan.")

    return hasil


# Template dimuat sekali saat modul diimpor pertama kali.
# Ini lebih efisien daripada membaca file setiap frame kamera.
_TEMPLATE_CACHE: dict = _muat_semua_template()


# ------------------------------------------------------------------ #
#  FUNGSI UTAMA: COCOKKAN TEMPLATE KE ROI                            #
# ------------------------------------------------------------------ #

def cocokkan_template(roi: np.ndarray) -> tuple:
    """
    Mencocokkan ROI kamera dengan semua template nominal yang tersedia.

    Menggunakan multi-scale matching: template dicoba dalam berbagai
    ukuran agar cocok meskipun uang ada di jarak berbeda dari kamera.

    Metode: cv2.TM_CCOEFF_NORMED
      → Menghitung korelasi silang ternormalisasi.
      → Tidak sensitif terhadap kecerahan absolut (cocok untuk berbagai cahaya).
      → Hasilnya 0.0–1.0; semakin tinggi = semakin cocok.

    Parameter:
        roi : numpy array BGR dari area uang di kamera

    Mengembalikan:
        tuple berisi 3 elemen:
          - nominal_terbaik (str) : label nominal dengan skor tertinggi,
                                    atau "Tidak ada template" jika tidak ada
          - skor_terbaik   (float): nilai skor tertinggi (0.0 – 1.0)
          - semua_skor     (dict) : { label_nominal: skor_terbaik }
                                    untuk ditampilkan di layar debug
    """
    # Jika tidak ada template yang dimuat, langsung kembalikan kosong
    if not _TEMPLATE_CACHE:
        return "Tidak ada template", 0.0, {}

    # Pastikan ROI valid
    if roi is None or roi.size == 0:
        return "Tidak ada template", 0.0, {}

    # Konversi ROI ke grayscale.
    # Template juga sudah grayscale → perbandingan fair tanpa pengaruh warna.
    if len(roi.shape) == 3:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray = roi.copy()

    # Tinggi dan lebar ROI — dipakai untuk membatasi ukuran template
    roi_h, roi_w = roi_gray.shape[:2]

    nominal_terbaik = "Tidak ada template"
    skor_terbaik    = 0.0
    semua_skor      = {}

    # ---- Iterasi setiap template ----------------------------------------- #
    for kode, (label, template_gray) in _TEMPLATE_CACHE.items():
        skor_terbaik_template = 0.0  # Skor terbaik untuk template ini saja

        # ---- Multi-scale: coba berbagai ukuran template ------------------- #
        for skala in SKALA_MULTI:
            # Hitung ukuran baru template setelah di-resize
            t_h, t_w = template_gray.shape[:2]
            ukuran_baru_w = int(t_w * skala)
            ukuran_baru_h = int(t_h * skala)

            # Template tidak boleh lebih besar dari ROI
            if ukuran_baru_w >= roi_w or ukuran_baru_h >= roi_h:
                continue

            # Template tidak boleh terlalu kecil (noise bisa mengalahkan)
            if ukuran_baru_w < 20 or ukuran_baru_h < 20:
                continue

            # Resize template ke ukuran yang dicoba
            template_resized = cv2.resize(
                template_gray,
                (ukuran_baru_w, ukuran_baru_h),
                interpolation=cv2.INTER_AREA
            )

            # Jalankan template matching
            # Hasilnya adalah "heatmap" seberapa cocok template di tiap posisi ROI
            hasil_match = cv2.matchTemplate(
                roi_gray,
                template_resized,
                cv2.TM_CCOEFF_NORMED
            )

            # Ambil nilai maksimum dari heatmap (posisi paling cocok)
            _, skor_max, _, _ = cv2.minMaxLoc(hasil_match)

            # Simpan skor tertinggi dari semua skala untuk template ini
            if skor_max > skor_terbaik_template:
                skor_terbaik_template = skor_max

        # Simpan skor akhir template ini ke semua_skor
        semua_skor[label] = round(skor_terbaik_template, 3)

        # Update skor global terbaik
        if skor_terbaik_template > skor_terbaik:
            skor_terbaik    = skor_terbaik_template
            nominal_terbaik = label

    # Jika skor terbaik di bawah minimum, anggap tidak ada yang cocok
    if skor_terbaik < SKOR_MINIMUM:
        return "Tidak yakin (template)", skor_terbaik, semua_skor

    return nominal_terbaik, round(skor_terbaik, 3), semua_skor
