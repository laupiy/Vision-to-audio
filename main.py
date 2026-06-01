"""
====================================================================
main.py — Orchestrator Utama Vision-to-Audio Bridge (WEB SERVER)
====================================================================
Versi v7.0 Multi-Device:
- Mode Server Camera: streaming kamera server via MJPEG
- Mode Device Camera: client mengirim frame dari kamera device,
  server memproses dan mengembalikan label + frame yang sudah dianotasi
- HTTPS agar getUserMedia bisa berjalan di device lain
- Listen pada 0.0.0.0 agar bisa diakses dari jaringan lokal
====================================================================
"""

import cv2
import time
import numpy as np
import threading
import base64
import ssl
import os
from flask import Flask, Response, request, jsonify, send_from_directory

from modules import config
from modules import roi_detector
from modules import color_detector
from modules import lighting
from modules import speech
from modules import ui
from modules import money_validator
from modules import template_matcher
from modules import hybrid_decision
from modules.preprocessing import preprocess_roi

app = Flask(__name__, static_folder='frontendweb', static_url_path='')

# ------------------------------------------------------------
# GLOBAL STATE
# ------------------------------------------------------------
camera = None
is_camera_on = False
mode_roi = "GUIDE"
current_label = "Kamera Mati"
latest_frame_encoded = None
state_lock = threading.Lock()

web_params = {
    'brightness': 50,
    'contrast': 50,
    'confidence': 75,
    'overlap': 30,
    'opacity': 80
}

STATUS_NON_NOMINAL = {
    "Letakkan uang di dalam kotak",
    "Arahkan uang ke kamera",
    "Cahaya Kurang",
    "Objek bukan uang",
    "Tidak terdeteksi",
    "Tidak yakin",
    "Kamera Mati"
}


def buka_kamera():
    kandidat_backend = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    kandidat_index   = [0, 1, 2]

    for backend in kandidat_backend:
        for index in kandidat_index:
            kam = cv2.VideoCapture(index, backend)
            if kam.isOpened():
                kam.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
                kam.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
                kam.set(cv2.CAP_PROP_FPS, 30)

                ret, _ = kam.read()
                if ret:
                    print(f"[INFO] Kamera aktif: index={index}, backend={backend}")
                    return kam
            kam.release()
    return None


def apply_web_adjustments(frame):
    brightness = web_params['brightness'] - 50
    contrast = web_params['contrast'] / 50.0
    b_val = brightness * 2.5
    frame_adj = cv2.convertScaleAbs(frame, alpha=contrast, beta=b_val)
    return frame_adj


def proses_roi_hybrid(roi: np.ndarray, metode_roi: str) -> tuple:
    if roi is None or roi.size == 0:
        return ("Arahkan uang ke kamera", "Arahkan uang ke kamera", "Tidak ada template", 0.0, "Tidak yakin")

    roi_processed = preprocess_roi(roi)
    roi_hsv = cv2.cvtColor(roi_processed, cv2.COLOR_BGR2HSV)

    if lighting.cek_kondisi_cahaya(roi_hsv):
        return ("Cahaya Kurang", "Cahaya Kurang", "Tidak ada template", 0.0, "Tidak yakin")

    if metode_roi == "AUTO" or metode_roi == "TRACKING":
        tekstur_valid = money_validator.validasi_tekstur_uang(roi_processed)
        warna_valid = money_validator.validasi_variasi_warna(roi_processed)
        if not tekstur_valid or not warna_valid:
            return ("Objek bukan uang", "Objek bukan uang", "Tidak ada template", 0.0, "Tidak yakin")
    elif metode_roi == "GUIDE" and config.VALIDASI_KETAT_GUIDE:
        tekstur_valid = money_validator.validasi_tekstur_uang(roi_processed)
        warna_valid = money_validator.validasi_variasi_warna(roi_processed)
        if not tekstur_valid or not warna_valid:
            return ("Objek bukan uang", "Objek bukan uang", "Tidak ada template", 0.0, "Tidak yakin")

    hasil_hsv = color_detector.tentukan_nominal(roi_processed)
    hasil_template, skor_template, _ = template_matcher.cocokkan_template(roi_processed)
    label_final, sumber = hybrid_decision.gabungkan_keputusan(hasil_hsv, hasil_template, skor_template)

    return label_final, hasil_hsv, hasil_template, skor_template, sumber


def process_single_frame(frame):
    """
    Proses 1 frame tunggal (dari server camera atau dari client device).
    Mengembalikan (label_final, frame_annotated).
    """
    frame = apply_web_adjustments(frame)
    frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
    frame_blur = cv2.medianBlur(frame, 5)

    roi = None
    bbox = None
    is_fallback = (mode_roi == "GUIDE")
    metode_tekstur = mode_roi

    if mode_roi == "GUIDE":
        hasil_roi = roi_detector.ambil_roi_guide(frame_blur)
        label_awal = "Letakkan uang di dalam kotak"
        if hasil_roi is not None:
            roi, bbox = hasil_roi
    else:
        label_awal = "Arahkan uang ke kamera"
        hasil_roi = roi_detector.cari_kotak_uang(frame_blur)
        if hasil_roi is not None:
            roi, bbox = hasil_roi
            metode_tekstur = "AUTO"

    if roi is None or bbox is None:
        label_final = label_awal
        hasil_hsv = label_awal
        hasil_template = "Tidak ada template"
        skor_template = 0.0
        sumber = "Tidak yakin"

        hasil_guide = roi_detector.ambil_roi_guide(frame_blur)
        if hasil_guide is not None:
            _, bbox = hasil_guide
            is_fallback = True
        else:
            bbox = (0, 0, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    else:
        label_final, hasil_hsv, hasil_template, skor_template, sumber = \
            proses_roi_hybrid(roi, metode_tekstur)

    # Draw overlays
    ui.gambar_bounding_box(frame, bbox, label_final, is_fallback=is_fallback)
    ui.gambar_hud(frame, label_final, 0.0)
    ui.gambar_info_hybrid(frame, hasil_hsv, hasil_template, skor_template, sumber, label_final)

    return label_final, frame


# ============================================================
# SERVER CAMERA LOOP (untuk mode kamera server)
# ============================================================
def camera_loop():
    global camera, is_camera_on, current_label, latest_frame_encoded, mode_roi

    tracker = None
    is_tracking = False
    counter_lost = 0
    waktu_fps_sebelumnya = time.time()
    fps_aktif = 0.0

    while True:
        if not is_camera_on:
            time.sleep(0.1)
            # Reset tracker saat kamera dimatikan
            tracker = None
            is_tracking = False
            counter_lost = 0
            continue

        if camera is None or not camera.isOpened():
            camera = buka_kamera()
            if camera is None:
                with state_lock:
                    current_label = "Gagal buka kamera"
                time.sleep(1)
                continue

        ret, frame = camera.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame = apply_web_adjustments(frame)
        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        frame_blur = cv2.medianBlur(frame, 5)

        roi = None
        bbox = None
        is_fallback = (mode_roi == "GUIDE")
        metode_tekstur = mode_roi

        if mode_roi == "GUIDE":
            hasil_roi = roi_detector.ambil_roi_guide(frame_blur)
            label_awal = "Letakkan uang di dalam kotak"
            if hasil_roi is not None:
                roi, bbox = hasil_roi
        else:
            label_awal = "Arahkan uang ke kamera"
            if not is_tracking:
                hasil_roi = roi_detector.cari_kotak_uang(frame_blur)
                if hasil_roi is not None:
                    roi, bbox = hasil_roi
                    metode_tekstur = "AUTO"
                    tracker = roi_detector.inisialisasi_tracker()
                    if tracker is not None:
                        success = tracker.init(frame_blur, bbox)
                        if success:
                            is_tracking = True
            else:
                success, tracked_bbox = tracker.update(frame_blur)
                if success:
                    bbox = tuple(map(int, tracked_bbox))
                    x, y, w, h = bbox
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                    roi = frame_blur[y1:y2, x1:x2]
                    metode_tekstur = "TRACKING"
                    counter_lost = 0
                else:
                    counter_lost += 1
                    if counter_lost > 5:
                        is_tracking = False
                        tracker = None

        if roi is None or bbox is None:
            label_final = label_awal
            hasil_hsv = label_awal
            hasil_template = "Tidak ada template"
            skor_template = 0.0
            sumber = "Tidak yakin"

            hasil_guide = roi_detector.ambil_roi_guide(frame_blur)
            if hasil_guide is not None:
                _, bbox = hasil_guide
                is_fallback = True
            else:
                bbox = (0, 0, config.FRAME_WIDTH, config.FRAME_HEIGHT)
        else:
            label_final, hasil_hsv, hasil_template, skor_template, sumber = \
                proses_roi_hybrid(roi, metode_tekstur)

        # Drawing
        ui.gambar_bounding_box(frame, bbox, label_final, is_fallback=is_fallback)
        ui.gambar_hud(frame, label_final, fps_aktif)
        ui.gambar_info_hybrid(frame, hasil_hsv, hasil_template, skor_template, sumber, label_final)

        waktu_sekarang = time.time()
        selang_waktu = waktu_sekarang - waktu_fps_sebelumnya
        if selang_waktu > 0:
            fps_aktif = 1.0 / selang_waktu
        waktu_fps_sebelumnya = waktu_sekarang

        with state_lock:
            if current_label != label_final and label_final not in STATUS_NON_NOMINAL:
                speech.bicara_nominal(label_final)

            current_label = label_final

            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if ret:
                latest_frame_encoded = buffer.tobytes()


def generate_frames():
    global latest_frame_encoded, is_camera_on
    while True:
        if not is_camera_on:
            time.sleep(0.1)
            continue

        with state_lock:
            frame_data = latest_frame_encoded

        if frame_data is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

        time.sleep(0.03)


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/')
def index():
    return send_from_directory('frontendweb', 'index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    with state_lock:
        return jsonify({'label': current_label if is_camera_on else "Kamera Mati"})

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global is_camera_on, camera, current_label
    data = request.json
    is_camera_on = data.get('state', False)

    if not is_camera_on and camera is not None:
        camera.release()
        camera = None
        with state_lock:
            current_label = "Kamera Mati"

    return jsonify({'success': True, 'state': is_camera_on})

@app.route('/update_params', methods=['POST'])
def update_params():
    global mode_roi, web_params
    data = request.json

    for key, val in data.items():
        if key in web_params:
            web_params[key] = float(val)
        if key == 'mode':
            mode_roi = val

    if 'confidence' in data:
        config.TEMPLATE_THRESHOLD_KUAT = float(data['confidence']) / 100.0

    return jsonify({'success': True})

@app.route('/replay_audio', methods=['POST'])
def replay_audio():
    with state_lock:
        if current_label not in STATUS_NON_NOMINAL:
            speech.bicara_nominal(current_label)
    return jsonify({'success': True})


# ============================================================
# DEVICE CAMERA ENDPOINT
# Client mengirim frame JPEG base64, server memproses, 
# mengembalikan label + frame yang sudah dianotasi
# ============================================================
@app.route('/process_client_frame', methods=['POST'])
def process_client_frame():
    global current_label
    try:
        data = request.json
        img_data = data.get('frame', '')

        # Decode base64 JPEG -> numpy array
        if ',' in img_data:
            img_data = img_data.split(',')[1]

        img_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'label': 'Frame error', 'frame': ''})

        label_final, frame_annotated = process_single_frame(frame)

        # Update global label & trigger TTS
        with state_lock:
            if current_label != label_final and label_final not in STATUS_NON_NOMINAL:
                speech.bicara_nominal(label_final)
            current_label = label_final

        # Encode annotated frame back to base64 JPEG
        ret, buffer = cv2.imencode('.jpg', frame_annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        if ret:
            encoded_frame = base64.b64encode(buffer).decode('utf-8')
        else:
            encoded_frame = ''

        return jsonify({
            'label': label_final,
            'frame': 'data:image/jpeg;base64,' + encoded_frame
        })

    except Exception as e:
        print(f"[ERROR] process_client_frame: {e}")
        return jsonify({'label': 'Error', 'frame': ''})


# Fallback untuk request lama
@app.route('/process_frame', methods=['GET', 'POST'])
def process_frame_fallback():
    return jsonify({'label': current_label, 'info': 'endpoint deprecated'})

# Static file catch-all — HARUS paling bawah
@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('frontendweb', path)


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    # Start the server camera loop thread
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

    # Cek apakah SSL cert tersedia
    cert_file = os.path.join(os.path.dirname(__file__), 'cert.pem')
    key_file  = os.path.join(os.path.dirname(__file__), 'key.pem')
    use_ssl = os.path.exists(cert_file) and os.path.exists(key_file)

    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print("======================================================")
    print(" Server Vision-to-Audio Aktif")
    print("------------------------------------------------------")
    if use_ssl:
        print(f" Lokal   : https://127.0.0.1:5000")
        print(f" Jaringan: https://{local_ip}:5000")
        print(" (Gunakan HTTPS agar kamera device bisa aktif)")
    else:
        print(f" Lokal   : http://127.0.0.1:5000")
        print(f" Jaringan: http://{local_ip}:5000")
        print(" [!] cert.pem / key.pem tidak ditemukan.")
        print("     Kamera device TIDAK akan berfungsi tanpa HTTPS.")
    print("======================================================")

    ssl_ctx = None
    if use_ssl:
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(cert_file, key_file)

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        use_reloader=False,
        ssl_context=ssl_ctx
    )