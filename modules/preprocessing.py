import cv2
import numpy as np


def gray_world_white_balance(image):
    """
    Menormalkan warna gambar.
    Berguna kalau cahaya kamera terlalu kuning, biru, atau redup.
    """
    result = image.copy().astype("float32")

    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])

    if avg_b == 0 or avg_g == 0 or avg_r == 0:
        return image

    avg_gray = (avg_b + avg_g + avg_r) / 3.0

    result[:, :, 0] *= avg_gray / avg_b
    result[:, :, 1] *= avg_gray / avg_g
    result[:, :, 2] *= avg_gray / avg_r

    result = np.clip(result, 0, 255).astype("uint8")
    return result


def gamma_correction(image, gamma=1.25):
    """
    Membantu mencerahkan gambar yang terlalu gelap.
    gamma > 1 membuat gambar lebih terang.
    """
    inv_gamma = 1.0 / gamma

    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype("uint8")

    return cv2.LUT(image, table)


def apply_clahe_hsv(image):
    """
    Meningkatkan kontras pencahayaan pada channel V di HSV.
    Ini membantu saat uang terlihat redup atau pucat.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    v_clahe = clahe.apply(v)

    hsv_result = cv2.merge([h, s, v_clahe])
    result = cv2.cvtColor(hsv_result, cv2.COLOR_HSV2BGR)

    return result


def boost_saturation(image, factor=1.20):
    """
    Menaikkan saturasi warna agar warna uang yang pucat lebih terlihat.
    Berguna untuk Rp20.000 yang hijau tapi sering terlihat abu-abu.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype("float32")

    hsv[:, :, 1] *= factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

    hsv = hsv.astype("uint8")
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return result


def denoise_image(image):
    """
    Mengurangi noise ringan.
    Kernel 3x3 dipakai agar detail angka uang tidak terlalu hilang.
    """
    return cv2.medianBlur(image, 3)


# ------------------------------------------------------------------ #
#  DETEKSI DIGITAL vs FISIK                                          #
# ------------------------------------------------------------------ #

def is_digital_source(roi):
    """
    Mendeteksi apakah ROI berasal dari layar digital (HP/monitor) atau
    uang fisik nyata.

    Layar digital memiliki ciri khas:
    - Saturasi rata-rata tinggi dan seragam (layar menghasilkan warna pekat)
    - Kontras/kecerahan sangat merata (backlight layar)
    - Noise rendah (piksel layar sangat rapi)

    Uang fisik ciri khas:
    - Saturasi bervariasi (tergantung pencahayaan ruangan)
    - Ada bayangan, lipatan, noise kamera
    - Warna lebih kusam/pucat dibanding digital

    Return: True jika kemungkinan besar dari layar digital
    """
    if roi is None or roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    mean_s = float(np.mean(s_channel))
    std_s = float(np.std(s_channel))
    mean_v = float(np.mean(v_channel))
    std_v = float(np.std(v_channel))

    # Layar digital: saturasi tinggi + seragam, kecerahan tinggi + seragam
    # Uang fisik: saturasi lebih rendah dan bervariasi
    is_high_saturation = mean_s > 80
    is_uniform_saturation = std_s < 45
    is_bright = mean_v > 130
    is_uniform_brightness = std_v < 50

    # Hitung skor digital (semakin tinggi = semakin yakin digital)
    score = 0
    if is_high_saturation:
        score += 1
    if is_uniform_saturation:
        score += 1
    if is_bright:
        score += 1
    if is_uniform_brightness:
        score += 1

    # Minimal 3 dari 4 ciri harus terpenuhi
    return score >= 3


# ------------------------------------------------------------------ #
#  PIPELINE PREPROCESSING UTAMA (ADAPTIF)                            #
# ------------------------------------------------------------------ #

def preprocess_roi(roi):
    """
    Pipeline preprocessing utama dengan deteksi adaptif.
    - Jika sumber digital: preprocessing ringan (sudah saturated)
    - Jika sumber fisik: preprocessing lebih kuat untuk normalisasi warna
    """
    processed = roi.copy()

    digital = is_digital_source(roi)

    if digital:
        # --- Mode Digital ---
        # Warna sudah bagus, hanya perlu sedikit normalisasi
        processed = gray_world_white_balance(processed)
        processed = gamma_correction(processed, gamma=1.15)  # Lebih ringan
        processed = apply_clahe_hsv(processed)
        # Saturation boost minimal karena sudah saturated
        processed = boost_saturation(processed, factor=1.05)
        processed = denoise_image(processed)
    else:
        # --- Mode Fisik ---
        # Warna pucat/kusam, perlu normalisasi lebih agresif
        # tapi JANGAN over-boost saturasi karena bisa shift hue

        # 1. White balance lebih kuat
        processed = gray_world_white_balance(processed)

        # 2. Gamma sedikit lebih tinggi untuk mencerahkan
        processed = gamma_correction(processed, gamma=1.35)

        # 3. CLAHE untuk meratakan pencahayaan
        processed = apply_clahe_hsv(processed)

        # 4. Saturasi boost MODERAT
        #    Terlalu tinggi akan menggeser hue dan merusak deteksi
        #    Terlalu rendah tidak akan membantu warna pucat
        processed = boost_saturation(processed, factor=1.30)

        # 5. Denoise sedikit lebih kuat untuk fisik (noise kamera)
        processed = denoise_image(processed)

    return processed


def preprocess_for_template(image):
    """
    Preprocessing khusus template matching.
    Fokusnya bukan warna, tapi bentuk angka/garis.
    Output berupa edge image dengan Canny Adaptive.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Perbaiki pencahayaan agar bentuk lebih menonjol
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    gray = clahe.apply(gray)

    # Blur dikurangi sedikit agar angka tidak hilang
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Canny Thresholding untuk kondisi kurang cahaya
    v = np.median(gray)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    edges = cv2.Canny(gray, lower, upper)

    # Dilation untuk mempertebal garis tepi agar pencocokan lebih toleran
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    return edges