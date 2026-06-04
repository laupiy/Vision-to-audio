"""
====================================================================
hybrid_decision.py — Penggabungan Keputusan HSV + Template Matching
====================================================================
Versi perbaikan v2:
- Lebih percaya template saat HSV tidak yakin (untuk uang fisik)
- Template SEDANG + HSV tidak yakin → percaya template (bukan reject)
- Saat konflik, pertimbangkan kedua skor sebelum memutuskan
- Tetap konservatif untuk Rp2.000 (false positive tinggi)
====================================================================
"""


# ------------------------------------------------------------------ #
#  KONFIGURASI THRESHOLD TEMPLATE                                    #
# ------------------------------------------------------------------ #

# matchTemplate TM_CCOEFF_NORMED: skor >= 0.55 dianggap kuat
THRESHOLD_TEMPLATE_KUAT = 0.55

# Skor >= 0.35 dianggap sedang (bisa konfirmasi HSV)
THRESHOLD_TEMPLATE_SEDANG = 0.35

# Threshold koreksi khusus Rp2.000 (abu-abu rawan false positive)
THRESHOLD_KOREKSI_RP2000 = 0.45

# Threshold baru: template sedang-kuat (bisa dipercaya saat HSV gagal)
THRESHOLD_TEMPLATE_SEDANG_KUAT = 0.42


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
    "Objek bukan uang",
    None,
}


# ------------------------------------------------------------------ #
#  HELPER                                                            #
# ------------------------------------------------------------------ #

def status_template(skor_template: float) -> str:
    if skor_template >= THRESHOLD_TEMPLATE_KUAT:
        return "kuat"
    if skor_template >= THRESHOLD_TEMPLATE_SEDANG:
        return "sedang"
    return "lemah"


def label_valid_nominal(label: str) -> bool:
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

    Prioritas keputusan (v2 - lebih percaya template):
    1. Jika keduanya sama → langsung percaya (sumber: HSV + Template cocok)
    2. Jika HSV tidak yakin tapi template kuat/sedang-kuat → pakai template
    3. Jika HSV yakin tapi template tidak tersedia → pakai HSV
       (kecuali Rp2.000 karena abu-abu rawan false positive)
    4. Jika konflik HSV vs Template:
       - Template kuat (>= 0.55) → pakai template
       - Template sedang-kuat (>= 0.42) → cek konteks, bisa pakai template
       - Template sedang (0.35-0.41) → tidak yakin
       - Template lemah → pakai HSV
    5. Rp2.000 dari HSV selalu butuh dukungan template (>= 0.45)
    """

    try:
        skor_template = float(skor_template)
    except (TypeError, ValueError):
        skor_template = 0.0

    template_tidak_yakin = hasil_template in LABEL_TIDAK_YAKIN_TEMPLATE
    hsv_tidak_yakin = hasil_hsv in LABEL_TIDAK_YAKIN_HSV
    status = status_template(skor_template)

    # -------------------------------------------------------------- #
    # 0. STRICT FILTER: Jika template yakin ini bukan uang           #
    # -------------------------------------------------------------- #
    if hasil_template == "Objek bukan uang":
        return "Objek bukan uang", "Template sangat rendah (Bukan uang)"

    # -------------------------------------------------------------- #
    # 1. Keduanya tidak yakin                                        #
    # -------------------------------------------------------------- #
    if hsv_tidak_yakin and template_tidak_yakin:
        return "Tidak yakin", "Keduanya tidak yakin"

    # -------------------------------------------------------------- #
    # 2. HSV dan Template sama → langsung percaya                    #
    # -------------------------------------------------------------- #
    if not hsv_tidak_yakin and not template_tidak_yakin:
        if hasil_hsv == hasil_template:
            # Sama-sama setuju → sangat yakin
            if hasil_hsv == "Rp2.000" and skor_template < THRESHOLD_KOREKSI_RP2000:
                # Rp2.000 tetap perlu threshold template cukup
                return "Tidak yakin", "Rp2.000 perlu konfirmasi template lebih kuat"
            return hasil_hsv, "HSV + Template cocok"

    # -------------------------------------------------------------- #
    # 3. Template tidak tersedia, andalkan HSV                       #
    # -------------------------------------------------------------- #
    if template_tidak_yakin:
        if hsv_tidak_yakin:
            return "Tidak yakin", "Tidak yakin"
        # Rp2.000 tidak boleh lolos tanpa template
        if hasil_hsv == "Rp2.000":
            return "Tidak yakin", "Rp2.000 butuh dukungan template"
        return hasil_hsv, "HSV"

    # -------------------------------------------------------------- #
    # 4. HSV tidak yakin, gunakan template jika cukup kuat           #
    #    PERBAIKAN: template sedang-kuat sekarang bisa dipercaya     #
    # -------------------------------------------------------------- #
    if hsv_tidak_yakin:
        if skor_template >= THRESHOLD_TEMPLATE_KUAT:
            return hasil_template, "Template kuat"
        # BARU: Template sedang-kuat (>= 0.42) sekarang dipercaya
        # Ini penting untuk uang fisik dimana HSV sering gagal
        # tapi template masih bisa mengenali pola uang
        if skor_template >= THRESHOLD_TEMPLATE_SEDANG_KUAT:
            # Rp2.000 tetap harus hati-hati
            if hasil_template == "Rp2.000":
                return "Tidak yakin", "Rp2.000 template sedang, perlu konfirmasi"
            return hasil_template, "Template sedang-kuat (HSV gagal)"
        if skor_template >= THRESHOLD_TEMPLATE_SEDANG:
            # Template sedang biasa, masih belum cukup yakin
            return "Tidak yakin", f"Template sedang, HSV tidak yakin"
        return "Tidak yakin", "Template lemah, HSV tidak yakin"

    # -------------------------------------------------------------- #
    # 5. Konflik: HSV yakin vs Template berbeda                      #
    # -------------------------------------------------------------- #
    if hasil_hsv != hasil_template:
        # Rp2.000 dari HSV: template bisa mengoreksi
        if hasil_hsv == "Rp2.000":
            if skor_template >= THRESHOLD_KOREKSI_RP2000:
                return hasil_template, "Template koreksi Rp2.000"
            return "Tidak yakin", "Rp2.000 HSV, template lemah"

        # Nominal lain: template kuat bisa menang
        if skor_template >= THRESHOLD_TEMPLATE_KUAT:
            return hasil_template, "Template kuat override HSV"

        # BARU: Template sedang-kuat + konflik → masih bisa percaya template
        # Karena untuk uang fisik, HSV bisa salah tapi template lebih stabil
        if skor_template >= THRESHOLD_TEMPLATE_SEDANG_KUAT:
            return hasil_template, "Template sedang-kuat override HSV"

        # Template sedang tapi berbeda: tidak yakin
        if skor_template >= THRESHOLD_TEMPLATE_SEDANG:
            return "Tidak yakin", "Konflik HSV vs Template sedang"

        # Template lemah: percaya HSV
        return hasil_hsv, "HSV diprioritaskan"

    # -------------------------------------------------------------- #
    # 6. Keduanya sama (sudah ditangani di atas, fallback)           #
    # -------------------------------------------------------------- #
    return hasil_hsv, "HSV"


# ------------------------------------------------------------------ #
#  FORMAT DEBUG                                                      #
# ------------------------------------------------------------------ #

def format_info_debug(
    hasil_hsv: str,
    hasil_template: str,
    skor_template: float,
    label_final: str,
    sumber: str
) -> list:
    try:
        skor_template = float(skor_template)
    except (TypeError, ValueError):
        skor_template = 0.0

    status = status_template(skor_template)

    baris = []

    baris.append((
        f"HSV     : {hasil_hsv}",
        (0, 255, 255)
    ))

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

    baris.append((
        f"Final   : {label_final}",
        (100, 255, 100) if label_valid_nominal(label_final) else (0, 165, 255)
    ))

    baris.append((
        f"Sumber  : {sumber}",
        (200, 200, 200)
    ))

    return baris
