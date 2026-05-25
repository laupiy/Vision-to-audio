"""
====================================================================
hybrid_decision.py — Penggabungan Keputusan HSV + Template Matching
====================================================================
Modul ini bertugas menggabungkan dua sumber informasi:
  1. Hasil deteksi HSV dari color_detector.py
  2. Hasil template matching dari template_matcher.py

Mengapa perlu digabungkan?
  → HSV cepat dan efisien, tapi sensitif terhadap pencahayaan.
    Contoh: Rp20.000 (hijau) bisa terlihat abu-abu di cahaya redup
    → HSV membacanya sebagai Rp2.000 (abu-abu).
  → Template matching mencari pola angka/gambar, bukan warna.
    Di kondisi cahaya sama, template masih bisa membaca "20000".
  → Dengan menggabungkan keduanya, sistem lebih tahan terhadap
    kondisi pencahayaan yang berubah-ubah.

Aturan keputusan:
  1. Jika skor template >= THRESHOLD_TEMPLATE_KUAT → pakai template.
  2. Jika skor template >= THRESHOLD_TEMPLATE_SEDANG dan ada konflik
     antara HSV dan template → pakai template (lebih terpercaya).
  3. Jika skor template < THRESHOLD_TEMPLATE_SEDANG → pakai HSV.
  4. Khusus: Jika HSV = "Rp2.000" tapi template = nominal lain dengan
     skor cukup → pakai template (koreksi false positive abu-abu).
  5. Jika keduanya tidak yakin → kembalikan "Tidak yakin".
====================================================================
"""


# ------------------------------------------------------------------ #
#  KONFIGURASI THRESHOLD                                             #
# ------------------------------------------------------------------ #

# Skor template di atas nilai ini dianggap sangat kuat → pakai template.
# Contoh: 0.65 artinya template cocok 65% atau lebih → percaya template.
THRESHOLD_TEMPLATE_KUAT = 0.65

# Skor template di atas nilai ini dianggap cukup untuk mengoreksi HSV.
# Khususnya dipakai saat ada konflik antara HSV dan template.
THRESHOLD_TEMPLATE_SEDANG = 0.45

# Label-label yang dianggap "tidak yakin" dari hasil HSV
LABEL_TIDAK_YAKIN_HSV = {
    "Tidak terdeteksi",
    "Tidak yakin",
    "Objek bukan uang",
    "Cahaya Kurang",
    "Letakkan uang di dalam kotak",
    "Arahkan uang ke kamera",
}

# Label yang keluar dari template_matcher saat tidak bisa menentukan
LABEL_TIDAK_YAKIN_TEMPLATE = {
    "Tidak ada template",
    "Tidak yakin (template)",
}


# ------------------------------------------------------------------ #
#  FUNGSI UTAMA: GABUNGKAN KEPUTUSAN                                 #
# ------------------------------------------------------------------ #

def gabungkan_keputusan(
    hasil_hsv      : str,
    hasil_template : str,
    skor_template  : float
) -> tuple:
    """
    Menggabungkan hasil HSV dan template matching menjadi satu keputusan final.

    Logika keputusan (urutan prioritas):

    [Kasus 1] Template tidak tersedia atau tidak ada template yang dimuat
      → Tidak ada template sama sekali → Pakai HSV saja.

    [Kasus 2] Skor template sangat kuat (>= THRESHOLD_TEMPLATE_KUAT)
      → Template sangat yakin → Pakai template tanpa melihat HSV.

    [Kasus 3] HSV = Rp2.000 tapi template mengatakan sesuatu yang lain
              dengan skor cukup (>= THRESHOLD_TEMPLATE_SEDANG)
      → Ini adalah koreksi false positive abu-abu.
      → Pakai template untuk mengoreksi.

    [Kasus 4] HSV tidak yakin tapi template cukup yakin
      → HSV menyerah, template masih ada sinyal → Pakai template.

    [Kasus 5] Template sedang tapi HSV juga punya hasil berbeda
      → Skor template tidak cukup kuat untuk mengoreksi HSV → Pakai HSV.

    [Kasus 6] Keduanya tidak yakin
      → Tidak ada yang bisa dipercaya → Kembalikan "Tidak yakin".

    Parameter:
        hasil_hsv      : string label dari color_detector.tentukan_nominal()
        hasil_template : string label dari template_matcher.cocokkan_template()
        skor_template  : float skor template (0.0 – 1.0)

    Mengembalikan:
        tuple (label_final, sumber_keputusan):
          - label_final       : string nominal akhir yang ditampilkan
          - sumber_keputusan  : "HSV", "Template", atau "Tidak yakin"
            (dipakai untuk tampilan debug di layar)
    """

    template_tidak_ada  = hasil_template in LABEL_TIDAK_YAKIN_TEMPLATE
    hsv_tidak_yakin     = hasil_hsv in LABEL_TIDAK_YAKIN_HSV

    # ---- Kasus 1: Tidak ada template sama sekali ---- #
    if template_tidak_ada:
        if hsv_tidak_yakin:
            return "Tidak yakin", "Tidak yakin"
        return hasil_hsv, "HSV"

    # ---- Kasus 2: Template sangat kuat ---- #
    # Skor sangat tinggi → percaya template sepenuhnya
    if skor_template >= THRESHOLD_TEMPLATE_KUAT:
        return hasil_template, "Template"

    # ---- Kasus 3: Koreksi false positive Rp2.000 ---- #
    # Rp2.000 abu-abu sering muncul sebagai false positive HSV
    # saat uang lain (hijau Rp20.000, dll.) terlihat pucat/abu-abu.
    # Jika template mendeteksi nominal LAIN dengan skor cukup,
    # template lebih terpercaya daripada HSV.
    if (hasil_hsv == "Rp2.000"
            and hasil_template != "Rp2.000"
            and skor_template >= THRESHOLD_TEMPLATE_SEDANG):
        return hasil_template, "Template (koreksi Rp2.000)"

    # ---- Kasus 4: HSV tidak yakin tapi template cukup yakin ---- #
    if hsv_tidak_yakin and skor_template >= THRESHOLD_TEMPLATE_SEDANG:
        return hasil_template, "Template"

    # ---- Kasus 5: Keduanya punya pendapat, template tidak cukup kuat ---- #
    # HSV punya jawaban dan template tidak cukup kuat untuk mengoreksi
    if not hsv_tidak_yakin:
        return hasil_hsv, "HSV"

    # ---- Kasus 6: Keduanya tidak yakin ---- #
    return "Tidak yakin", "Tidak yakin"


# ------------------------------------------------------------------ #
#  FUNGSI HELPER: FORMAT TAMPILAN DEBUG                              #
# ------------------------------------------------------------------ #

def format_info_debug(
    hasil_hsv      : str,
    hasil_template : str,
    skor_template  : float,
    label_final    : str,
    sumber         : str
) -> list:
    """
    Menghasilkan daftar baris teks untuk ditampilkan di layar debug.

    Teks ini menunjukkan perbandingan antara hasil HSV, template,
    dan keputusan akhir hybrid, sehingga memudahkan penjelasan
    cara kerja sistem kepada dosen atau penguji.

    Parameter:
        hasil_hsv      : label dari HSV
        hasil_template : label dari template matching
        skor_template  : skor template (float 0.0–1.0)
        label_final    : keputusan akhir
        sumber         : siapa yang menang ("HSV" / "Template" / dst)

    Mengembalikan:
        list of tuple: [(teks, warna_bgr), ...]
        Setiap tuple berisi teks dan warna untuk cv2.putText().
    """
    baris = []

    # Baris 1: Hasil HSV
    baris.append((
        f"HSV     : {hasil_hsv}",
        (0, 255, 255)   # Kuning
    ))

    # Baris 2: Hasil Template + Skor
    warna_template = (100, 255, 100) if skor_template >= THRESHOLD_TEMPLATE_SEDANG else (150, 150, 150)
    baris.append((
        f"Template: {hasil_template} [{skor_template:.2f}]",
        warna_template
    ))

    # Baris 3: Sumber keputusan
    baris.append((
        f"Sumber  : {sumber}",
        (200, 200, 200)   # Abu-abu terang
    ))

    return baris
