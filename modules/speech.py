"""
====================================================================
speech.py — Text-to-Speech Non-Blocking dengan Cooldown Cerdas
====================================================================
Masalah pada kode asli (main.py versi awal):
  - pyttsx3.runAndWait() bersifat BLOCKING: seluruh proses Python
    berhenti hingga pembacaan selesai.
  - Akibatnya, frame kamera "macet" / freeze selama ~2–3 detik saat
    nominal sedang dibacakan.

Solusi yang diterapkan di modul ini:
  - Menggunakan variabel state internal (_label_sebelumnya, _hitung_stabil,
    _waktu_suara_terakhir) yang disimpan sebagai atribut modul.
  - Fungsi bicara_nominal() hanya memanggil engine.say() +
    engine.runAndWait() jika SEMUA syarat terpenuhi:
      1. Label bukan "Tidak terdeteksi" / "Tidak yakin" / "Cahaya Kurang"
      2. Label stabil (tidak berubah selama minimal FRAME_STABIL frame)
      3. Waktu saat ini sudah melewati COOLDOWN_SUARA sejak suara terakhir

Catatan:
  Fungsi ini masih BLOCKING saat eksekusi TTS karena keterbatasan
  pyttsx3. Solusi benar-benar non-blocking membutuhkan threading atau
  subprocess, yang ada di rencana pengembangan di README.md.
====================================================================
"""

import time
import pyttsx3
from . import config


# ------------------------------------------------------------------ #
#  INISIALISASI TTS ENGINE (Singleton Module-Level)                  #
# ------------------------------------------------------------------ #

# Inisialisasi engine sekali saja saat modul di-import.
# Tidak perlu dibuat ulang di setiap loop frame karena ini mahal (slow).
_engine = pyttsx3.init()

# Kecepatan bicara: 150 kata/menit (default 200, diperlambat agar jelas)
_engine.setProperty("rate",   150)

# Volume: 1.0 = maksimum
_engine.setProperty("volume", 1.0)


# ------------------------------------------------------------------ #
#  STATE INTERNAL MODUL (Pengganti Variabel Global di main.py)       #
# ------------------------------------------------------------------ #

# Label nominal yang dideteksi pada iterasi sebelumnya.
# Digunakan untuk melacak stabilitas label antar-frame.
_label_sebelumnya: str  = ""

# Penghitung jumlah frame berturut-turut dengan label yang sama.
# Suara hanya dibacakan jika penghitung ini mencapai FRAME_STABIL.
_hitung_stabil: int     = 0

# Timestamp (detik) terakhir kali suara diputar.
# Mencegah TTS berbicara terus-menerus ketika uang diam di depan kamera.
_waktu_suara_terakhir: float = 0.0

# Jumlah frame berturut-turut yang sama sebelum suara diizinkan diputar.
# Nilai lebih tinggi = lebih lambat bereaksi tapi lebih stabil.
FRAME_STABIL: int = 5

# Label-label yang TIDAK boleh dibacakan oleh TTS
LABEL_TIDAK_VALID = {"Tidak terdeteksi", "Tidak yakin", "Cahaya Kurang"}


# ------------------------------------------------------------------ #
#  FUNGSI UTAMA                                                      #
# ------------------------------------------------------------------ #

def bicara_nominal(label_aktif: str) -> bool:
    """
    Mencoba membacakan label nominal jika semua syarat terpenuhi.

    Fungsi ini dipanggil di setiap iterasi loop utama dengan label
    yang saat ini aktif. Fungsi akan MENOLAK membunyikan suara jika:
      - label termasuk dalam LABEL_TIDAK_VALID
      - label baru berubah (reset penghitung stabilitas)
      - label belum stabil selama FRAME_STABIL frame berturut-turut
      - belum lewat COOLDOWN_SUARA detik sejak suara terakhir

    Parameter:
        label_aktif : string label nominal saat ini (dari color_detector atau
                      override dari lighting/ui)

    Mengembalikan:
        True  → Suara berhasil diputar pada pemanggilan ini
        False → Suara tidak diputar (syarat belum terpenuhi)
    """
    global _label_sebelumnya, _hitung_stabil, _waktu_suara_terakhir

    # ---- Syarat 1: Label harus valid (bukan status error/kosong) ---------- #
    if label_aktif in LABEL_TIDAK_VALID:
        # Reset stabilitas sehingga label valid berikutnya mulai dari nol
        _label_sebelumnya = ""
        _hitung_stabil    = 0
        return False

    # ---- Syarat 2: Stabilitas label antar-frame -------------------------- #
    if label_aktif == _label_sebelumnya:
        # Label sama dengan frame sebelumnya → naikkan penghitung
        _hitung_stabil += 1
    else:
        # Label berbeda → reset penghitung dari awal
        _label_sebelumnya = label_aktif
        _hitung_stabil    = 1
        return False   # Belum stabil, tidak perlu cek waktu

    # Jika penghitung belum mencapai threshold stabilitas, tunggu
    if _hitung_stabil < FRAME_STABIL:
        return False

    # ---- Syarat 3: Cooldown waktu antar-pembacaan ----------------------- #
    waktu_sekarang = time.time()
    if waktu_sekarang - _waktu_suara_terakhir < config.COOLDOWN_SUARA:
        return False   # Terlalu cepat sejak pembacaan terakhir

    # ---- Semua syarat terpenuhi: Cari teks suara dan bacakan ------------- #
    # Import di sini untuk menghindari circular import
    from .color_detector import ambil_label_suara
    teks_suara = ambil_label_suara(label_aktif)

    print(f"[TTS] Membacakan: {teks_suara}")

    # engine.say() + runAndWait() bersifat blocking selama pembacaan.
    # Ini menyebabkan frame kamera berhenti sementara — lihat README.md
    # untuk rencana perbaikan menggunakan threading.
    _engine.say(teks_suara)
    _engine.runAndWait()

    # Catat waktu pembacaan terakhir setelah runAndWait selesai
    _waktu_suara_terakhir = time.time()

    return True


def reset_state() -> None:
    """
    Mereset semua state internal modul speech ke kondisi awal.

    Berguna saat program perlu memulai ulang sesi deteksi tanpa
    menutup dan membuka ulang aplikasi.
    """
    global _label_sebelumnya, _hitung_stabil, _waktu_suara_terakhir
    _label_sebelumnya     = ""
    _hitung_stabil        = 0
    _waktu_suara_terakhir = 0.0
