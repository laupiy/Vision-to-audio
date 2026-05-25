"""
====================================================================
ambil_dataset_hsv.py — Skrip Kolektor Dataset Warna (Ukuran Kotak Jumbo)
====================================================================
"""
import cv2
import numpy as np

def main():
    kamera = cv2.VideoCapture(0)
    
    print("=" * 50)
    print(" SKRIP KOLEKTOR DATASET WARNA (KOTAK JUMBO)")
    print("=" * 50)
    nama_nominal = input("Masukkan nama nominal sampel (Contoh: 50rb): ").strip()
    
    nama_file_dataset = "dataset_warna.txt"
    print(f"\n[INFO] Data akan otomatis disimpan ke dalam file: {nama_file_dataset}")
    print("[INFO] Tekan 's' untuk MENYIMPAN sampel warna ke dataset.")
    print("[INFO] Tekan 'q' untuk selesai.")
    print("=" * 50)

    count_sampel = 0

    while True:
        ret, frame = kamera.read()
        if not ret: break
        
        frame = cv2.resize(frame, (640, 480))
        h, w = frame.shape[:2]
        
        # ── KOTAK DIPERBESAR (Mengambil 80% Lebar dan 70% Tinggi Layar) ──
        x1, y1 = int(w * 0.1), int(h * 0.15)
        x2, y2 = int(w * 0.9), int(h * 0.85)
        
        display_frame = frame.copy()
        # Menggunakan warna hijau terang agar lebih nyaman dilihat saat kalibrasi
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.putText(display_frame, f"Sampel {nama_nominal}: {count_sampel} terekam", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, "Posisikan uang di dalam kotak hijau", (15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        cv2.imshow("Kolektor Dataset Warna", display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            # Ambil ROI berdasarkan ukuran kotak baru yang lebih luas
            roi = frame[y1:y2, x1:x2]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            h_ch, s_ch, v_ch = hsv_roi[:,:,0], hsv_roi[:,:,1], hsv_roi[:,:,2]
            
            # Persentil diubah menjadi 3% dan 97% agar mencakup spektrum warna tepi uang yang lebih luas
            lower = [int(np.percentile(h_ch, 3)), int(np.percentile(s_ch, 3)), int(np.percentile(v_ch, 3))]
            upper = [int(np.percentile(h_ch, 97)), int(np.percentile(s_ch, 97)), int(np.percentile(v_ch, 97))]
            
            count_sampel += 1
            
            # Tulis langsung ke file txt di dalam proyek
            with open(nama_file_dataset, "a") as f:
                f.write(f"Nominal: {nama_nominal} | Sampel ke-{count_sampel} | Lower: {lower} | Upper: {upper}\n")
                
            print(f"[SUKSES] Sampel ke-{count_sampel} untuk {nama_nominal} disimpan ke proyek.")
            
        elif key == ord('q'):
            break

    kamera.release()
    cv2.destroyAllWindows()
    print(f"\n[SELESAI] Silakan cek file '{nama_file_dataset}' di folder proyekmu!")

if __name__ == "__main__":
    main()