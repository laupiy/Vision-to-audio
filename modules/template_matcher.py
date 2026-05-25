import cv2
import numpy as np
from pathlib import Path

from modules.preprocessing import preprocess_for_template


TEMPLATE_DIR = Path("templates")

NOMINAL_LABELS = {
    "100000": "Rp100.000",
    "50000": "Rp50.000",
    "20000": "Rp20.000",
    "10000": "Rp10.000",
    "5000": "Rp5.000",
    "2000": "Rp2.000",
    "1000": "Rp1.000",
}


def crop_area_angka(roi):
    """
    Mengambil area angka nominal dari ROI.
    Untuk uang landscape, angka nominal biasanya ada di kiri atas.
    """
    h, w = roi.shape[:2]

    x1 = 0
    y1 = 0
    x2 = int(w * 0.50)
    y2 = int(h * 0.50)

    area_angka = roi[y1:y2, x1:x2]

    return area_angka


def load_templates():
    """
    Membaca template dari folder templates.
    Mendukung png, jpg, dan jpeg.
    Nama file harus berupa nominal:
    20000.jpg berarti Rp20.000
    """
    templates = []

    extensions = ["*.png", "*.jpg", "*.jpeg"]

    for ext in extensions:
        for path in TEMPLATE_DIR.glob(ext):
            key = path.stem

            if key not in NOMINAL_LABELS:
                continue

            image = cv2.imread(str(path))

            if image is None:
                print(f"[WARNING] Template gagal dibaca: {path}")
                continue

            processed_template = preprocess_for_template(image)

            templates.append({
                "key": key,
                "label": NOMINAL_LABELS[key],
                "path": path,
                "image": processed_template,
            })

    return templates


def match_template_nominal(roi):
    """
    Mencocokkan area angka nominal dengan template angka.
    Return:
    - label terbaik
    - skor terbaik
    - semua skor
    - status kekuatan skor
    """
    templates = load_templates()

    if len(templates) == 0:
        return {
            "label": "Tidak ada template",
            "score": 0.0,
            "status": "lemah",
            "all_scores": []
        }

    # Crop area angka dari ROI
    area_angka = crop_area_angka(roi)

    # Preprocessing ROI angka
    roi_processed = preprocess_for_template(area_angka)

    best_label = "Tidak yakin"
    best_score = 0.0
    all_scores = []

    for item in templates:
        template = item["image"]

        # Hindari template lebih besar daripada ROI
        if template.shape[0] > roi_processed.shape[0] or template.shape[1] > roi_processed.shape[1]:
            continue

        # Multi-scale sederhana
        scale_scores = []

        for scale in np.linspace(0.65, 1.25, 13):
            new_w = int(template.shape[1] * scale)
            new_h = int(template.shape[0] * scale)

            if new_w <= 5 or new_h <= 5:
                continue

            resized_template = cv2.resize(template, (new_w, new_h))

            if resized_template.shape[0] > roi_processed.shape[0] or resized_template.shape[1] > roi_processed.shape[1]:
                continue

            result = cv2.matchTemplate(
                roi_processed,
                resized_template,
                cv2.TM_CCOEFF_NORMED
            )

            _, max_val, _, _ = cv2.minMaxLoc(result)
            scale_scores.append(max_val)

        if len(scale_scores) == 0:
            score = 0.0
        else:
            score = float(max(scale_scores))

        all_scores.append((item["label"], score))

        if score > best_score:
            best_score = score
            best_label = item["label"]

    if best_score >= 0.78:
        status = "kuat"
    elif best_score >= 0.68:
        status = "sedang"
    else:
        status = "lemah"

    all_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)

    return {
        "label": best_label,
        "score": best_score,
        "status": status,
        "all_scores": all_scores
    }
def cocokkan_template(roi):
    """
    Wrapper agar kompatibel dengan main.py lama.

    main.py memanggil:
        template_matcher.cocokkan_template(roi)

    Sedangkan versi baru memakai:
        match_template_nominal(roi)

    Fungsi ini menyamakan format return:
        hasil_template, skor_template, semua_skor
    """
    hasil = match_template_nominal(roi)

    label = hasil.get("label", "Tidak ada template")
    score = hasil.get("score", 0.0)
    all_scores = hasil.get("all_scores", [])

    return label, score, all_scores