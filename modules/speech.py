"""
====================================================================
speech.py — Text-to-Speech dengan Capture & Validation System
====================================================================
Perbaikan v5:
  - BARU: Capture Window — mengumpulkan N frame deteksi sebelum
    memutuskan output audio
  - BARU: Majority Voting — dari N frame, ambil nominal yang paling
    sering muncul. Minimal X% frame harus setuju.
  - BARU: Anti-collision — audio baru tidak bisa dimulai selama
    audio lama masih jalan
  - Menggunakan threading.Thread agar engine.runAndWait() tidak lagi
    membuat frame kamera macet / freeze.
====================================================================
"""

import queue
import time
import threading
import pyttsx3
from collections import Counter
from . import config

# ------------------------------------------------------------------ #
#  TTS BACKGROUND THREAD (Persistent)                                #
# ------------------------------------------------------------------ #
# Menggunakan satu thread persisten dan Queue untuk menghindari
# error pyttsx3 / COM (SAPI5) di Windows saat dipanggil dari banyak thread berbeda.
_tts_queue = queue.Queue()
_tts_sedang_bicara = False

def _tts_worker():
    global _tts_sedang_bicara, _waktu_suara_terakhir
    
    # Inisialisasi engine di dalam thread worker (sangat penting untuk Windows)
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 1.0)
    
    while True:
        teks_suara = _tts_queue.get()
        if teks_suara is None:
            break
            
        try:
            _tts_sedang_bicara = True
            engine.say(teks_suara)
            engine.runAndWait()
        except Exception as e:
            print(f"[WARN] Error pada TTS engine: {e}")
        finally:
            _waktu_suara_terakhir = time.time()
            _tts_sedang_bicara = False
            _tts_queue.task_done()

# Mulai thread persisten
_thread_suara = threading.Thread(target=_tts_worker, daemon=True)
_thread_suara.start()


# ------------------------------------------------------------------ #
#  STATE INTERNAL MODUL                                              #
# ------------------------------------------------------------------ #
_waktu_suara_terakhir: float = 0.0

# Capture window: buffer yang mengumpulkan hasil deteksi
_capture_buffer: list = []
_capture_active: bool = False
_capture_start_time: float = 0.0
_capture_lock = threading.Lock()

# Label yang tidak valid (status bukan nominal uang)
LABEL_TIDAK_VALID = {
    "Tidak terdeteksi", "Tidak yakin", "Cahaya Kurang",
    "Letakkan uang di dalam kotak", "Arahkan uang ke kamera",
    "Objek bukan uang"
}


# (Worker thread sudah dipindah ke atas)


# ------------------------------------------------------------------ #
#  FUNGSI CAPTURE WINDOW                                             #
# ------------------------------------------------------------------ #

def _start_capture():
    """Memulai capture window baru."""
    global _capture_buffer, _capture_active, _capture_start_time
    _capture_buffer = []
    _capture_active = True
    _capture_start_time = time.time()


def _add_to_capture(label: str):
    """Menambahkan label deteksi ke capture buffer."""
    global _capture_buffer
    _capture_buffer.append(label)


def _evaluate_capture() -> str:
    """
    Mengevaluasi capture buffer menggunakan majority voting.

    Return:
        Nama nominal jika konsensus tercapai, atau None jika gagal.
    """
    global _capture_buffer

    if not _capture_buffer:
        return None

    # Hitung frekuensi setiap label (hanya yang valid)
    valid_labels = [l for l in _capture_buffer if l not in LABEL_TIDAK_VALID]

    if not valid_labels:
        return None

    counter = Counter(valid_labels)
    total_valid = len(valid_labels)
    total_semua = len(_capture_buffer)

    # Ambil yang paling sering muncul
    most_common_label, most_common_count = counter.most_common(1)[0]

    # Syarat 1: Label paling sering harus minimal CAPTURE_CONSENSUS_PERSEN dari total frame
    consensus_persen = (most_common_count / total_semua) * 100.0
    if consensus_persen < config.CAPTURE_CONSENSUS_PERSEN:
        print(f"[CAPTURE] Konsensus gagal: {most_common_label} hanya {consensus_persen:.0f}% "
              f"(butuh {config.CAPTURE_CONSENSUS_PERSEN}%)")
        return None

    # Syarat 2: Harus ada minimal 3 frame valid
    if most_common_count < 3:
        print(f"[CAPTURE] Terlalu sedikit frame valid: {most_common_count}")
        return None

    print(f"[CAPTURE] Konsensus tercapai: {most_common_label} "
          f"({most_common_count}/{total_semua} = {consensus_persen:.0f}%)")
    return most_common_label


def _end_capture():
    """Mengakhiri capture window."""
    global _capture_active, _capture_buffer
    _capture_active = False
    _capture_buffer = []


# ------------------------------------------------------------------ #
#  FUNGSI UTAMA (BARU - DENGAN CAPTURE VALIDATION)                   #
# ------------------------------------------------------------------ #

def bicara_nominal(label_aktif: str) -> bool:
    """
    Sistem capture + validasi sebelum output audio.

    Alur kerja:
    1. Jika label valid dan belum ada capture aktif → mulai capture window
    2. Selama capture aktif → kumpulkan label ke buffer
    3. Setelah buffer penuh (CAPTURE_WINDOW_SIZE frame) → evaluasi
    4. Jika konsensus tercapai → output audio
    5. Jika gagal → buang buffer, mulai ulang
    """
    global _waktu_suara_terakhir, _tts_sedang_bicara
    global _capture_active, _capture_buffer

    # Jika thread suara sebelumnya masih berbicara, abaikan
    if _tts_sedang_bicara:
        return False

    # Cooldown setelah suara terakhir
    waktu_sekarang = time.time()
    if waktu_sekarang - _waktu_suara_terakhir < config.CAPTURE_COOLDOWN:
        return False

    with _capture_lock:
        # ---- Label tidak valid: reset capture ---- #
        if label_aktif in LABEL_TIDAK_VALID:
            if _capture_active:
                print(f"[CAPTURE] Dibatalkan — label tidak valid: {label_aktif}")
                _end_capture()
            return False

        # ---- Mulai capture baru jika belum aktif ---- #
        if not _capture_active:
            _start_capture()
            _add_to_capture(label_aktif)
            print(f"[CAPTURE] Mulai mengumpulkan — frame 1/{config.CAPTURE_WINDOW_SIZE}: {label_aktif}")
            return False

        # ---- Capture sedang berjalan: tambahkan ke buffer ---- #
        _add_to_capture(label_aktif)
        frame_count = len(_capture_buffer)
        print(f"[CAPTURE] Frame {frame_count}/{config.CAPTURE_WINDOW_SIZE}: {label_aktif}")

        # ---- Buffer belum penuh: lanjutkan kumpulkan ---- #
        if frame_count < config.CAPTURE_WINDOW_SIZE:
            return False

        # ---- Buffer penuh: evaluasi konsensus ---- #
        hasil = _evaluate_capture()
        _end_capture()

        if hasil is None:
            print(f"[CAPTURE] Tidak ada konsensus — audio dibatalkan")
            return False

        # ---- Konsensus tercapai: output audio ---- #
        from .color_detector import ambil_label_suara
        teks_suara = ambil_label_suara(hasil)

        print(f"[TTS THREAD] Mengirim ke antrean: {teks_suara}")

        # Masukkan ke queue untuk diproses oleh thread TTS tunggal
        _tts_queue.put(teks_suara)

        return True


def reset_state() -> None:
    """Mereset semua state internal modul speech ke kondisi awal."""
    global _waktu_suara_terakhir, _tts_sedang_bicara
    global _capture_active, _capture_buffer
    _waktu_suara_terakhir = 0.0
    _tts_sedang_bicara    = False
    with _capture_lock:
        _capture_active = False
        _capture_buffer = []