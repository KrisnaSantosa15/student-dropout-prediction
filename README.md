# Proyek Akhir: Menyelesaikan Permasalahan Jaya Jaya Institut

- Nama: Krisna Santosa
- Email: mamang.krisna15@gmail.com
- Id Dicoding: [krisna_santosa](https://www.dicoding.com/users/krisna_santosa)


## Business Understanding

Jaya Jaya Institut menghadapi angka dropout siswa yang cukup tinggi. Kondisi ini berdampak pada reputasi institusi, efektivitas proses pembelajaran, serta perencanaan akademik. Tim manajemen membutuhkan pendekatan berbasis data untuk:

1. Memahami faktor yang paling terkait dengan dropout.
2. Mengidentifikasi segmen siswa berisiko lebih dini.
3. Memonitor indikator penting melalui dashboard.
4. Membangun sistem prediksi sebagai early warning.

### Permasalahan Bisnis

1. Faktor apa saja yang paling berkaitan dengan status dropout siswa?
2. Kelompok siswa mana yang memiliki risiko dropout tertinggi?
3. Bagaimana institusi dapat memonitor risiko dropout secara periodik?
4. Dapatkah dibangun model machine learning untuk memprediksi risiko dropout sejak awal?

### Cakupan Proyek

1. Pengambilan data students' performance dari sumber resmi Dicoding.
2. Data understanding mencakup pemeriksaan struktur data, kualitas data, missing values, duplikasi, dan distribusi target.
3. EDA: univariate, bivariate, dan multivariate untuk menemukan pola dropout.
4. Data preparation dengan penanganan missing values dan duplikasi data.
5. Transformasi target menjadi klasifikasi biner: `Dropout` vs `Non-Dropout`.
6. Pelatihan beberapa model klasifikasi dan pemilihan model terbaik.
7. Penyimpanan model dan metadata untuk kebutuhan inferensi.
8. Pembuatan data agregasi untuk dashboard monitoring.
9. Implementasi aplikasi dashboard + prediksi menggunakan Streamlit.

### Persiapan

Sumber data:
1. Dataset utama: [Students' Performance Dataset](https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/main/students_performance/data.csv)
2. Deskripsi dataset: [README Dataset](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md)

Setup environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

Melatih model dan menyiapkan data dashboard:

```bash
python train_model.py
```

Prediksi batch dari CSV:

```bash
python prediction.py --input sample_prediction_input.csv --output prediction_output.csv
```

Notebook analisis ada di `notebook.ipynb` untuk eksplorasi data dan insight EDA.

Dashboard Streamlit:

```bash
streamlit run app.py
```


## Business Dashboard

Dashboard dibangun dengan Streamlit (`app.py`) agar tim akademik dapat melihat metrik dropout utama dan pola risikonya secara interaktif.

Fitur utama dashboard:
1. KPI utama: total siswa, jumlah dropout, dropout rate.
2. Distribusi dropout berdasarkan `Course`, `Gender`, dan status pembayaran (`Tuition_fees_up_to_date`).
3. Segmentasi umur (`Age Group`) untuk memonitor risiko per kelompok usia.
4. Tabel siswa berisiko tertinggi berdasarkan skor model prediksi.


## Menjalankan Sistem Machine Learning

Langkah menjalankan aplikasi (lokal):

```bash
streamlit run app.py
```

Prediksi batch dari CSV:

```bash
python prediction.py --input sample_prediction_input.csv --output prediction_output.csv
```

Output prediksi akan menambahkan kolom:
1. `dropout_prediction` (0/1)
2. `dropout_risk_score` (probabilitas)
3. `dropout_risk_label` (`High Risk`/`Low Risk`)


## Link Streamlit Cloud
[https://student-dropout-prediction-dicoding.streamlit.app/](https://student-dropout-prediction-dicoding.streamlit.app/)


## Conclusion

1. Proyek berhasil membangun alur data science end-to-end: data understanding, EDA terstruktur, preprocessing, modeling, evaluasi, dan prototipe deployment.
2. Penanganan missing values dilakukan dalam pipeline (`median` untuk numerik, `most_frequent` untuk kategorikal), sehingga data siap pakai untuk inferensi tanpa ketergantungan pembersihan manual.
3. Model terbaik yang terpilih adalah `RandomForest` dengan performa uji:
	- Accuracy: 88.81%
	- Precision: 86.85%
	- Recall: 76.76%
	- F1-score: 81.50%
	- ROC-AUC: 93.15%
4. Faktor-faktor yang paling berkaitan dengan dropout pada data ini adalah:
	- Status pembayaran kuliah (`Tuition_fees_up_to_date`): siswa dengan status pembayaran tidak up to date memiliki risiko dropout yang lebih tinggi.
	- Performa akademik semester awal: nilai `Curricular_units_1st_sem_grade` dan `Curricular_units_2nd_sem_grade` yang lebih rendah berkaitan kuat dengan peningkatan risiko dropout.
	- Program studi (`Course`): dropout rate tidak merata antar program studi, dengan beberapa program memiliki tingkat risiko jauh lebih tinggi.
	- Demografi tertentu (contoh gender dan kelompok usia): terdapat perbedaan dropout rate antarkelompok, sehingga intervensi perlu berbasis segmen.
5. Karakteristik umum siswa dropout yang teridentifikasi:
	- Lebih sering muncul pada siswa dengan skor akademik semester 1 dan 2 yang rendah.
	- Proporsinya lebih tinggi pada siswa dengan status pembayaran kuliah bermasalah.
	- Cenderung terkonsentrasi pada kelompok program studi berisiko tinggi.
6. Kombinasi dashboard interaktif + prediksi risiko dapat difungsikan sebagai early warning system untuk intervensi akademik yang lebih cepat dan terukur.

### Rekomendasi Action Items

1. Bangun program intervensi khusus untuk siswa dengan `dropout_risk_score` tertinggi (mentoring akademik, konseling, monitoring kehadiran).
2. Prioritaskan tindak lanjut pada segmen dengan dropout rate tinggi di tingkat `Course`.
3. Lakukan validasi bulanan terhadap performa model dan retraining berkala bila data terbaru sudah tersedia.
4. Integrasikan hasil prediksi ke SOP monitoring akademik agar setiap siswa berisiko memiliki rencana tindak lanjut yang jelas.
5. Tambahkan evaluasi fairness model lintas gender dan kelompok usia sebelum digunakan sebagai dasar kebijakan yang lebih luas.
