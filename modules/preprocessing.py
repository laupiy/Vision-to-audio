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


def preprocess_roi(roi):
    """
    Pipeline preprocessing utama.
    Dipakai sebelum HSV detection dan template matching.
    """
    processed = roi.copy()

    processed = gray_world_white_balance(processed)
    processed = gamma_correction(processed, gamma=1.25)
    processed = apply_clahe_hsv(processed)
    processed = boost_saturation(processed, factor=1.20)
    processed = denoise_image(processed)

    return processed


def preprocess_for_template(image):
    """
    Preprocessing khusus template matching.
    Fokusnya bukan warna, tapi bentuk angka/garis.
    Output berupa edge image.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray, 50, 150)

    return edges