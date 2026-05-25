"""
====================================================================
buat_template.py — Alat Bantu Pembuatan Template Nominal Uang
====================================================================
Script ini membantu Anda membuat template gambar nominal uang
yang diperlukan oleh modules/template_matcher.py.

Cara pakai:
  1. Jalankan: python buat_template.py
  2. Arahkan uang ke kamera
  3. Tekan SPASI untuk mengambil screenshot ROI area guide
  4. Tekan angka yang sesuai dengan nominal uang:
     1 = Rp1.000
     2 = Rp2.000
     5 = Rp5.000
     0 = Rp10.000 (tombol 0)
     Kemudian gunakan menu di layar untuk nominal besar
  5. File akan disimpan di folder templates/
  6. Tekan Q untuk keluar

Tips agar template akurat:
  - Pastikan uang rata dan tidak terlipat
  - Cahaya harus cukup terang dan merata
  - Letakkan di background polos (bukan meja ramai)
  - Ambil beberapa foto dari sudut sedikit berbeda
  - Gunakan foto terbaik sebagai template
====================================================================
"""

import cv2
import os
import time

# Konfigurasi guide (sama seperti di config.py)
GUIDE_X_RATIO = 0.08
GUIDE_Y_RATIO = 0.26
GUIDE_W_RATIO = 0.84
GUIDE_H_RATIO = 0.48

FRAME_WIDTH  = 640
FRAME_HEIGHT = 480

# Daftar nominal dan nama file output
DAFTAR_NOMINAL = {
    "1": ("Rp1.000",   "1000"),
    "2": ("Rp2.000",   "2000"),
    "5": ("Rp5.000",   "5000"),
    "0": ("Rp10.000",  "10000"),
    "r": ("Rp20.000",  "20000"),
    "f": ("Rp50.000",  "50000"),
    "s": ("Rp100.000", "100000"),
}

os.makedirs("templates", exist_ok=True)


def ambil_roi_guide(frame):
    h, w = frame.shape[:2]
    x1 = int(w * GUIDE_X_RATIO)
    y1 = int(h * GUIDE_Y_RATIO)
    gw = int(w * GUIDE_W_RATIO)
    gh = int(h * GUIDE_H_RATIO)
    x2 = min(w, x1 + gw)
    y2 = min(h, y1 + gh)
    return frame[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def main():
    print("=" * 55)
    print("  Alat Bantu Pembuatan Template Uang Rupiah")
    print("=" * 55)
    print("Tekan tombol di layar setelah uang berada di kotak:")
    for tombol, (label, _) in DAFTAR_NOMINAL.items():
        print(f"  [{tombol.upper()}] = {label}")
    print("  [Q]       = Keluar")
    print()

    kamera = None
    for idx in [0, 1, 2]:
        k = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if k.isOpened():
            k.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
            k.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            ret, _ = k.read()
            if ret:
                kamera = k
                break
        k.release()

    if kamera is None:
        for idx in [0, 1, 2]:
            k = cv2.VideoCapture(idx)
            if k.isOpened():
                ret, _ = k.read()
                if ret:
                    kamera = k
                    break
            k.release()

    if kamera is None:
        print("[ERROR] Kamera tidak ditemukan.")
        return

    status_pesan = "Arahkan uang ke dalam kotak, lalu tekan tombol nominal"
    waktu_pesan  = 0.0

    while True:
        ret, frame = kamera.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        roi, bbox = ambil_roi_guide(frame)
        x, y, w, h = bbox

        # Gambar kotak guide
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 255), 2)
        cv2.putText(frame, "Area Template", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # Petunjuk di bawah
        panduan = "1:1rb 2:2rb 5:5rb 0:10rb R:20rb F:50rb S:100rb | Q:Keluar"
        cv2.putText(frame, panduan, (10, FRAME_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        # Tampilkan pesan status setelah simpan
        if time.time() - waktu_pesan < 3.0:
            cv2.putText(frame, status_pesan, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

        cv2.imshow("Buat Template — Vision-to-Audio Bridge", frame)

        tombol = cv2.waitKey(1) & 0xFF

        if tombol == ord("q"):
            break

        # Simpan template jika tombol nominal ditekan
        tombol_chr = chr(tombol).lower()
        if tombol_chr in DAFTAR_NOMINAL:
            label, kode_file = DAFTAR_NOMINAL[tombol_chr]
            path_simpan      = os.path.join("templates", f"{kode_file}.png")

            # Simpan ROI sebagai file PNG
            cv2.imwrite(path_simpan, roi)

            status_pesan = f"DISIMPAN: {label} → templates/{kode_file}.png"
            waktu_pesan  = time.time()
            print(f"[OK] {status_pesan}")

    kamera.release()
    cv2.destroyAllWindows()

    # Tampilkan daftar template yang ada
    print("\n=== Template yang tersedia di folder templates/ ===")
    for f in sorted(os.listdir("templates")):
        if f.endswith((".png", ".jpg")):
            path = os.path.join("templates", f)
            img  = cv2.imread(path)
            if img is not None:
                print(f"  {f} ({img.shape[1]}x{img.shape[0]}px)")
    print("Selesai.")


if __name__ == "__main__":
    main()
