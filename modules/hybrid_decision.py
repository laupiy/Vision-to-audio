"""
====================================================================
hybrid_decision.py — Penggabungan Keputusan HSV + Template Matching
====================================================================
Modul ini menggabungkan dua hasil deteksi:
1. HSV Color Detection
2. Template Matching angka nominal

Perbaikan versi ini:
- Threshold template dinaikkan agar skor lemah tidak langsung menang.
- Template skor 0.57 tidak boleh mengubah hasil final.
- Rp2.000 dari HSV tidak langsung dipercaya karena abu-abu sering false positive.
- Jika HSV dan Template berbeda tapi skor template belum kuat, hasil menjadi "Tidak yakin".
====================================================================
"""


# ------------------------------------------------------------------ #
#  KONFIGURASI THRESHOLD TEMPLATE                                    #
# ------------------------------------------------------------------ #

# Template baru dianggap kuat jika skor sangat tinggi.
# Template kuat boleh menjadi hasil final.
THRESHOLD_TEMPLATE_KUAT = 0.78

# Template sedang hanya dianggap sinyal bantuan.
# Jika berbeda dengan HSV, jangan langsung dipakai.
THRESHOLD_TEMPLATE_SEDANG = 0.68

# Khusus untuk koreksi HSV Rp2.000.
# Rp2.000 sering false positive saat cahaya redup,
# jadi template boleh mengoreksi Rp2.000 jika skornya cukup kuat.
THRESHOLD_KOREKSI_RP2000 = 0.72


# ------------------------------------------------------------------ #
#  LABEL STATUS                                                       #
# ------------------------------------------------------------------ #

LABEL_TIDAK_YAKIN_HSV = {
    "Tidak terdeteksi",
    "Tidak yakin",
    "Objek bukan uang",
    "Cahaya Kurang",
    "Cahaya kurang",
    "Letakkan uang di dalam kotak",
    "Arahkan uang ke kamera",
    None,
}

LABEL_TIDAK_YAKIN_TEMPLATE = {
    "Tidak ada template",
    "Tidak yakin (template)",
    "Tidak yakin",
    "Template lemah",
    None,
}


# ------------------------------------------------------------------ #
#  HELPER                                                            #
# ------------------------------------------------------------------ #

def status_template(skor_template: float) -> str:
    """
    Mengubah skor template menjadi status:
    - kuat
    - sedang
    - lemah
    """
    if skor_template >= THRESHOLD_TEMPLATE_KUAT:
        return "kuat"

    if skor_template >= THRESHOLD_TEMPLATE_SEDANG:
        return "sedang"

    return "lemah"


def label_valid_nominal(label: str) -> bool:
    """
    Mengecek apakah label adalah nominal uang, bukan status error.
    """
    return label not in LABEL_TIDAK_YAKIN_HSV and label not in LABEL_TIDAK_YAKIN_TEMPLATE


# ------------------------------------------------------------------ #
#  FUNGSI UTAMA                                                      #
# ------------------------------------------------------------------ #

def gabungkan_keputusan(
    hasil_hsv: str,
    hasil_template: str,
    skor_template: float
) -> tuple:
    """
    Menggabungkan hasil HSV dan Template Matching.

    Aturan utama:
    1. Jika template kuat >= 0.78, template boleh menjadi final.
    2. Jika template sedang 0.68 - 0.77 dan sama dengan HSV, hasil dianggap yakin.
    3. Jika template sedang tapi berbeda dengan HSV, hasil "Tidak yakin".
    4. Jika template lemah < 0.68, jangan gunakan template.
    5. Khusus HSV Rp2.000:
       - Rp2.000 sering false positive karena abu-abu.
       - Jika template mengarah ke nominal lain dengan skor >= 0.72,
         gunakan template.
       - Jika template lemah, jangan langsung percaya Rp2.000.
    """

    # Pastikan skor berupa float
    try:
        skor_template = float(skor_template)
    except (TypeError, ValueError):
        skor_template = 0.0

    template_tidak_yakin = hasil_template in LABEL_TIDAK_YAKIN_TEMPLATE
    hsv_tidak_yakin = hasil_hsv in LABEL_TIDAK_YAKIN_HSV

    status = status_template(skor_template)

    # -------------------------------------------------------------- #
    # 1. Template tidak tersedia / tidak yakin                       #
    # -------------------------------------------------------------- #
    if template_tidak_yakin:
        if hsv_tidak_yakin:
            return "Tidak yakin", "Tidak yakin"

        # Khusus Rp2.000: jangan langsung percaya kalau template tidak ada
        if hasil_hsv == "Rp2.000":
            return "Tidak yakin", "HSV Rp2.000 tanpa dukungan template"

        return hasil_hsv, "HSV"

    # -------------------------------------------------------------- #
    # 2. HSV tidak yakin, template kuat                              #
    # -------------------------------------------------------------- #
    if hsv_tidak_yakin:
        if skor_template >= THRESHOLD_TEMPLATE_KUAT:
            return hasil_template, "Template kuat"

        return "Tidak yakin", f"Template {status}, HSV tidak yakin"

    # -------------------------------------------------------------- #
    # 3. HSV dan Template sama                                       #
    # -------------------------------------------------------------- #
    if hasil_hsv == hasil_template:
        if skor_template >= THRESHOLD_TEMPLATE_SEDANG:
            return hasil_hsv, "HSV + Template cocok"

        # Kalau sama tapi template masih lemah, tetap boleh pakai HSV
        # kecuali Rp2.000 karena rawan false positive
        if hasil_hsv == "Rp2.000":
            return "Tidak yakin", "Rp2.000 belum cukup didukung template"

        return hasil_hsv, "HSV"

    # -------------------------------------------------------------- #
    # 4. Template sangat kuat, boleh mengalahkan HSV                 #
    # -------------------------------------------------------------- #
    if skor_template >= THRESHOLD_TEMPLATE_KUAT:
        return hasil_template, "Template kuat"

    # -------------------------------------------------------------- #
    # 5. Kasus khusus: HSV Rp2.000                                   #
    # -------------------------------------------------------------- #
    if hasil_hsv == "Rp2.000":
        if hasil_template != "Rp2.000" and skor_template >= THRESHOLD_KOREKSI_RP2000:
            return hasil_template, "Template koreksi Rp2.000"

        return "Tidak yakin", "HSV Rp2.000 rawan false positive"

    # -------------------------------------------------------------- #
    # 6. Template sedang tapi berbeda dengan HSV                     #
    # -------------------------------------------------------------- #
    if skor_template >= THRESHOLD_TEMPLATE_SEDANG:
        return "Tidak yakin", "HSV dan Template berbeda"

    # -------------------------------------------------------------- #
    # 7. Template lemah, pakai HSV                                   #
    # -------------------------------------------------------------- #
    return hasil_hsv, "HSV"


# ------------------------------------------------------------------ #
#  FORMAT DEBUG UNTUK TAMPILAN                                       #
# ------------------------------------------------------------------ #

def format_info_debug(
    hasil_hsv: str,
    hasil_template: str,
    skor_template: float,
    label_final: str,
    sumber: str
) -> list:
    """
    Menghasilkan daftar teks debug untuk ditampilkan di layar.

    Return:
        list of tuple:
        [
            ("teks", warna_bgr),
            ...
        ]
    """

    try:
        skor_template = float(skor_template)
    except (TypeError, ValueError):
        skor_template = 0.0

    status = status_template(skor_template)

    baris = []

    # Baris HSV
    baris.append((
        f"HSV     : {hasil_hsv}",
        (0, 255, 255)
    ))

    # Baris Template
    if status == "kuat":
        warna_template = (0, 255, 0)
    elif status == "sedang":
        warna_template = (0, 165, 255)
    else:
        warna_template = (120, 120, 120)

    baris.append((
        f"Template: {hasil_template} [{skor_template:.2f}] ({status})",
        warna_template
    ))

    # Baris Final
    baris.append((
        f"Final   : {label_final}",
        (100, 255, 100) if label_valid_nominal(label_final) else (0, 165, 255)
    ))

    # Baris sumber keputusan
    baris.append((
        f"Sumber  : {sumber}",
        (200, 200, 200)
    ))

    return baris