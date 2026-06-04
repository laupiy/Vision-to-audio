Proyek **Vision-to-Audio Bridge** merupakan sebuah proyek akhir yang dirancang untuk mendeteksi dan mengenali nominal uang kertas Rupiah secara langsung melalui kamera, yang kemudian hasilnya dikonversi menjadi suara (Text-to-Speech).

Proyek akhir ini dilatarbelakangi oleh tantangan di lapangan saat membedakan nominal uang kertas yang seringkali memiliki warna memudar, berada dalam kondisi lecek, atau berada pada lingkungan dengan pencahayaan yang kurang baik. Sistem prototipe ini dibuat secara khusus untuk memberikan aksesibilitas yang lebih baik, khususnya bagi pengguna tunanetra, agar dapat mengenali nominal uang Rupiah secara mandiri dan akurat.

## Anggota Kelompok
- Putri Yudi P (152024035)
- Nuraeni Solihah (152024130)
- Syahrina Alma F (152024142)
- Deden Roga N (152024152)

## Rumusan Masalah
Pengembangan proyek akhir ini difokuskan untuk menyelesaikan beberapa masalah utama:
1. Bagaimana cara mengenali nominal uang Rupiah menggunakan kamera.
2. Bagaimana cara memproses citra uang agar lebih stabil terhadap berbagai kondisi pencahayaan.
3. Bagaimana cara membedakan nominal uang dengan mengekstraksi fitur warna dan angka nominal.
4. Bagaimana cara mengurangi kesalahan deteksi akibat warna uang yang pucat (misal: Rp20.000 yang memudar).
5. Bagaimana cara merubah hasil deteksi akhir menjadi output suara yang informatif.

## Arsitektur dan Metode Sistem
Sistem ini menggunakan gabungan teknik Pengolahan Citra Digital (Digital Image Processing) dan pemrosesan suara, yang terdiri dari tahapan berikut:

1. **ROI (Region of Interest) Guide**: Area panduan statis pada layar kamera untuk meletakkan uang agar deteksi lebih fokus (digunakan karena deteksi kontur otomatis masih belum stabil akibat background yang bervariasi).
2. **Preprocessing Citra**: Tahap penstabilan gambar untuk meminimalkan pengaruh pencahayaan dan memperjelas warna uang sebelum masuk ke tahap deteksi. Metode yang digunakan:
   - *Gray World White Balance*: Menormalkan warna gambar jika cahaya terlalu kuning, biru, atau tidak seimbang.
   - *Gamma Correction*: Membantu mencerahkan gambar yang terlalu gelap.
   - *CLAHE (Contrast Limited Adaptive Histogram Equalization)*: Diterapkan pada channel V (Value) di format HSV untuk meningkatkan kontras pencahayaan.
   - *Saturation Boost*: Memperjelas warna uang yang terlihat pucat.
   - *Median Blur*: Mengurangi noise ringan pada gambar citra.
3. **HSV Color Segmentation (Ekstraksi Fitur Warna)**: Mengonversi format awal citra (BGR) ke HSV untuk memudahkan pemisahan warna dominan dari tiap-tiap uang kertas.
4. **Template Matching (Ekstraksi Fitur Angka)**: Mencocokkan area pola angka pada uang dengan *template crop* angka (misal: 100000, 50000) yang sudah disimpan di dalam folder `templates/`.
5. **Hybrid Decision (Klasifikasi)**: Menggabungkan hasil dari HSV Color Detection dan Template Matching. Sistem membandingkan skor. Jika skor *template* tinggi, hasil *template* diprioritaskan. Hal ini sangat berguna untuk mengoreksi bias pada HSV (contoh: jika warna uang Rp20.000 memudar dan terdeteksi sebagai Rp2.000 oleh HSV, sistem akan mengoreksinya menjadi Rp20.000 jika pola angkanya sangat cocok).
6. **Text-to-Speech (TTS)**: Menggunakan library `pyttsx3` untuk mengonversi hasil klasifikasi dari nominal akhir ke dalam bentuk suara (contoh output: "Dua puluh ribu rupiah").

## Nominal Uang yang Didukung
Sistem mengenali 7 pecahan uang kertas Rupiah berdasarkan ekstraksi warna dominan:
- **Rp100.000** (Merah / Pink)
- **Rp50.000** (Biru)
- **Rp20.000** (Hijau)
- **Rp10.000** (Ungu)
- **Rp5.000** (Coklat / Kuning)
- **Rp2.000** (Abu-abu)
- **Rp1.000** (Hijau Kebiruan)

## Hasil Pengujian & Evaluasi
Berdasarkan hasil pengujian lapangan pada kondisi cahaya yang cukup, sistem dapat melakukan deteksi dengan status berhasil. Namun, pada kondisi cahaya redup, sistem masih perlu mendapatkan *tuning* lanjutan terutama untuk mendeteksi uang Rp20.000 yang sering terlihat pucat dan salah dikenali sebagai Rp2.000.

**Rencana Pengembangan Kedepannya:**
- Menambah ragam dataset uang dalam berbagai kondisi pencahayaan.
- Memperbaiki kualitas *template crop* angka nominal agar area potongannya lebih akurat.
- Mengimplementasikan metode *ORB Feature Matching* untuk pencocokan fitur yang jauh lebih handal dibandingkan *Template Matching*.
- Mengembangkan algoritma deteksi *Region of Interest* (ROI) otomatis secara stabil tanpa guide.
- Menerapkan arsitektur *Machine Learning* atau *Convolutional Neural Networks (CNN)* untuk versi lanjutan yang memiliki tingkat akurasi tinggi.
README.md
Menampilkan README.md.
