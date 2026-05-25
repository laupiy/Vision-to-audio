# Vision-to-Audio Bridge

Vision-to-Audio Bridge adalah project Pengolahan Citra Digital untuk mengenali nominal uang kertas Rupiah menggunakan kamera. Sistem ini menggunakan OpenCV untuk pengolahan citra, HSV Color Segmentation untuk deteksi warna, Template Matching untuk membantu mengenali angka nominal, dan pyttsx3 untuk membacakan hasil deteksi.

---

## Fitur

- Deteksi uang Rupiah melalui kamera.
- ROI Guide untuk meletakkan uang di dalam kotak panduan.
- Preprocessing agar gambar lebih stabil terhadap pencahayaan.
- Deteksi warna menggunakan HSV.
- Template Matching untuk mencocokkan angka nominal.
- Hybrid Decision untuk menggabungkan hasil HSV dan template.
- Output hasil berupa teks dan suara.

---

## Nominal yang Didukung

- Rp100.000
- Rp50.000
- Rp20.000
- Rp10.000
- Rp5.000
- Rp2.000
- Rp1.000

---

## Instalasi

Clone repository:

```bash
git clone https://github.com/laupiy/Vision-to-audio.git
cd Vision-to-audio
pip install -r requirements.txt
