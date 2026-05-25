"""
====================================================================
speech.py — Text-to-Speech TRUE NON-BLOCKING dengan Threading
====================================================================
Perbaikan v4:
  - Menggunakan threading.Thread agar engine.runAndWait() tidak lagi 
    membuat frame kamera macet / freeze.
  - Mengamankan state dengan lock sederhana agar tidak terjadi 
    tumpang tindih suara (deadlock pyttsx3).
====================================================================
"""

import time
import pyttsx3
import threading
from . import config


# ------------------------------------------------------------------ #
#  INISIALISASI TTS ENGINE (Singleton Module-Level)                  #
# ------------------------------------------------------------------ #
_engine = pyttsx3.init()
_engine.setProperty("rate",   150)
_engine.setProperty("volume", 1.0)

# Flag thread untuk memastikan hanya ada satu suara yang diputar dalam satu waktu
_tts_sedang_bicara = False


# ------------------------------------------------------------------ #
#  STATE INTERNAL MODUL                                              #
# ------------------------------------------------------------------ #
_label_sebelumnya: str  = ""
_hitung_stabil: int     = 0
_waktu_suara_terakhir: float = 0.0

FRAME_STABIL: int = 5
LABEL_TIDAK_VALID = {"Tidak terdeteksi", "Tidak yakin", "Cahaya Kurang", "Letakkan uang di dalam kotak", "Arahkan uang ke kamera", "Objek bukan uang"}


# ------------------------------------------------------------------ #
#  FUNGSI WORKER THREAD (BACKGROUND EXECUTION)                       #
# ------------------------------------------------------------------ #
def _worker_bicara(teks_suara):
    """Fungsi pembantu yang berjalan di background thread agar tidak memacetkan kamera"""
    global _tts_sedang_bicara, _waktu_suara_terakhir
    try:
        _tts_sedang_bicara = True
        _engine.say(teks_suara)
        _engine.runAndWait()
    except Exception as e:
        print(f"[WARN] Error pada background TTS thread: {e}")
    finally:
        _waktu_suara_terakhir = time.time()
        _tts_sedang_bicara = False


# ------------------------------------------------------------------ #
#  FUNGSI UTAMA                                                      #
# ------------------------------------------------------------------ #
def bicara_nominal(label_aktif: str) -> bool:
    """
    Mencoba membacakan label nominal menggunakan Threading (Non-Blocking).
    """
    global _label_sebelumnya, _hitung_stabil, _waktu_suara_terakhir, _tts_sedang_bicara

    # Jika thread suara sebelumnya masih berbicara, abaikan request frame ini (jangan tumpang tindih)
    if _tts_sedang_bicara:
        return False

    # ---- Syarat 1: Label harus valid ---------- #
    if label_aktif in LABEL_TIDAK_VALID:
        _label_sebelumnya = ""
        _hitung_stabil    = 0
        return False

    # ---- Syarat 2: Stabilitas label antar-frame ---------- #
    if label_aktif == _label_sebelumnya:
        _hitung_stabil += 1
    else:
        _label_sebelumnya = label_aktif
        _hitung_stabil    = 1
        return False

    if _hitung_stabil < FRAME_STABIL:
        return False

    # ---- Syarat 3: Cooldown waktu antar-pembacaan ---------- #
    waktu_sekarang = time.time()
    if waktu_sekarang - _waktu_suara_terakhir < config.COOLDOWN_SUARA:
        return False

    # ---- Semua syarat terpenuhi: Jalankan Background Thread ------------- #
    from .color_detector import ambil_label_suara
    teks_suara = ambil_label_suara(label_aktif)

    print(f"[TTS THREAD] Membacakan: {teks_suara}")

    # Membuat dan menjalankan thread baru khusus untuk suara
    thread_suara = threading.Thread(target=_worker_bicara, args=(teks_suara,), daemon=True)
    thread_suara.start()

    return True


def reset_state() -> None:
    """Mereset semua state internal modul speech ke kondisi awal."""
    global _label_sebelumnya, _hitung_stabil, _waktu_suara_terakhir, _tts_sedang_bicara
    _label_sebelumnya     = ""
    _hitung_stabil        = 0
    _waktu_suara_terakhir = 0.0
    _tts_sedang_bicara    = False