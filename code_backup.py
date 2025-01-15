import streamlit as st
import pandas as pd
import json
import re
import string
from nltk.tokenize import word_tokenize as token_kata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os
import numpy as np
import nltk
nltk.download('punkt')

# Pengaturan untuk halaman web steamlit
st.set_page_config(page_title="ABSA-KMeans", page_icon="💻")

# Streamlit application
st.title("ANALISIS FAKTOR-FAKTOR YANG MEMPENGARUHI PERPINDAHAN KARIR DENGAN PEMANFAATAN ASPECT-BASED SENTIMENT ANALYSIS MENGGUNAKAN METODE K-MEANS")
page = st.sidebar.selectbox("Tentukan Halaman:", ["Preprosesing", "Klastering", "Analisis Sentimen", "Visualisasi Data"])

# Data awal
file_path = "csv/look1.csv"  # Lokasi file CSV data awal
with open(file_path, "r", encoding="utf-8") as f:
    csv_raw_data = f.read()
    
    

# KUMPULAN FUNGSI PREPROSESING

# Fungsi untuk mengubah teks menjadi huruf kecil (lowercase)
def ubah_ke_huruf_kecil(dataframe, column_name):
    dataframe[column_name] = dataframe[column_name].str.lower()
    return dataframe

# Fungsi untuk menghapus karakte-karakter spesial twitter(X) dari data hasil scrapping
def bersihkan_karakter_twitter(text):
    text = text.replace('\\t', " ").replace('\\n', " ").replace('\\u', " ").replace('\\', "")
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = ' '.join(re.sub(r"([@#][A-Za-z0-9_]+)|(\w+:\/\/\S+)", " ", text).split())
    return text.replace("http://", " ").replace("https://", " ")

# Fungsi untuk menghapus angka untuk penyederhanaan data clear
def hapus_angka(text):
    return re.sub(r"\d+", "", text)

# Fungsi untuk menghapus tanda baca seperti titik, koma, tanda seru, dll.
def hapus_tandabaca(text):
    return text.translate(str.maketrans("", "", string.punctuation))

# Fungsi untuk menghapus whitespace atau spasi yang ada di awal dan akhir teks.
def hapus_spasi_awalakhir(text):
    return text.strip()

# Fungsi untuk mengganti spasi berturut turut dengan satu spasi tunggal.
def ganti_spasi_tunggal(text):
    return re.sub(r'\s+', ' ', text)

# Fungsi untuk menghapus karakter tunggal yang berdiri sendiri di dalam teks
def hapus_karakter_tunggal(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

# Fungsi untuk membaca file JSON yang berisi kamus slang (bahasa gaul) dan menyimpannya ke dalam bentuk dictionary
def muat_kamus_slang(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
# Fungsi untuk menggantikan bahasa gaul di dalam teks dengan padanan yang lebih baku berdasarkan kamus slang.
def slang_ke_baku(text, kamus_bahasa_gaul):
    return " ".join(kamus_bahasa_gaul.get(word, word) for word in text.split())

# Fungsi untuk memecah teks menjadi token atau kata kata individual
def bungkus_tokenisasi_kata(text):
    return token_kata(text)


class StopWordsIndo:
    def __init__(self, stopwords_file):
        self.stopwords = self.olah_stopword(stopwords_file)
    
    def olah_stopword(self, stopwords_file):
        with open(stopwords_file, 'r', encoding='utf-8') as f:
            return set(f.read().splitlines())
    
    def hapus_stopwords(self, text):
        pisahkan_kata = text.split()
        kata_bersih = [word for word in pisahkan_kata if word not in self.stopwords and len(word) > 3]
        return " ".join(kata_bersih)
class KamusFilter:
    def __init__(self, kamus_file):
        self.term_dict = self.baca_kamus(kamus_file)

    def baca_kamus(self, kamus_file):
        try:
            with open(kamus_file, 'r', encoding='utf-8') as file:
                return set(file.read().splitlines())
        except FileNotFoundError:
            print(f"File {kamus_file} tidak ditemukan.")
            return set()

    def hapus_bukan_id(self, document):
        return [term for term in document if term in self.term_dict]

# Stemming
pengolahdata = StemmerFactory() # Digunakan untuk membuat objek stemmer.
stemmer = pengolahdata.create_stemmer() # Membuat objek stemmer yang dapat digunakan untuk melakukan proses stemming.

# Fungsi untuk mengubah setiap kata ke bentuk dasarnya.
def sederhanakan_teks(text):
    return stemmer.stem(text)

# Fungsi untuk memuat leksikon sentimen
def load_lexicon(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return set(file.read().splitlines())    

# Preproses data untuk menjalankan fungsi yang sudah dibuat
def preprocess_data(uploaded_file):
    corpus_df = pd.read_csv(uploaded_file)
    corpus_df = ubah_ke_huruf_kecil(corpus_df, 'full_text')
    corpus_df['full_text'] = corpus_df['full_text'].apply(bersihkan_karakter_twitter)
    corpus_df['full_text'] = corpus_df['full_text'].apply(hapus_angka)
    corpus_df['full_text'] = corpus_df['full_text'].apply(hapus_tandabaca)
    corpus_df['full_text'] = corpus_df['full_text'].apply(hapus_spasi_awalakhir)
    corpus_df['full_text'] = corpus_df['full_text'].apply(ganti_spasi_tunggal)
    corpus_df['full_text'] = corpus_df['full_text'].apply(hapus_karakter_tunggal)

    # Load dan replace slang
    kamus_bahasa_gaul = muat_kamus_slang("txt/kamusSlang.json")
    corpus_df['full_text'] = corpus_df['full_text'].apply(lambda x: slang_ke_baku(x, kamus_bahasa_gaul))

    # Tokenisasi
    corpus_df['tokenisasi'] = corpus_df['full_text'].apply(bungkus_tokenisasi_kata)

    # Stemming
    corpus_df['stemmed'] = corpus_df['full_text'].apply(sederhanakan_teks)

    # Inisialisasi stopword dan kamus filter
    stopwords_processor = StopWordsIndo('txt/stopwords.txt')
    kamus_filter = KamusFilter("txt/kamusIndonesia.txt")

    # Hapus stopword
    corpus_df['stopwords'] = corpus_df['stemmed'].apply(lambda x: stopwords_processor.hapus_stopwords(x))

    # Filter term non-indonesia
    corpus_df['filtered'] = corpus_df['stopwords'].apply(lambda x: kamus_filter.hapus_bukan_id(x.split()))

    # Menyimpan hasil preproses kedalam file 'hasil.txt'
    corpus_df.to_csv('preprosesing/hasil.txt', index=None, header=True)
    return corpus_df


# Logika aplikasi
if page == "Preprosesing":
    st.header("Persiapan Data")
    
    # Tombol Download Data Mentah
    st.write("Unduh dataset untuk melanjutkan proses analisis di bawah")
    st.download_button(
        label="Download File CSV",
        data=csv_raw_data,
        file_name="dataset.csv",
        mime="text/csv",
        use_container_width= True,
    )
    
    uploaded_file = st.file_uploader("Pilih file CSV", type='csv')

    # Memeriksa apakah file ada dalam session_state
    if 'uploaded_file' in st.session_state:
        st.write("File yang sedang diproses ...")
        st.write(st.session_state.uploaded_file.name)  # Menampilkan nama file yang diupload

    if uploaded_file is not None:
        # Menyimpan file ke dalam session_state untuk akses di semua halaman tanpa upload ulang
        st.session_state.uploaded_file = uploaded_file

        # Memastikan nama file hanya disimpan satu kali
        if 'uploaded_file_name' not in st.session_state:
            # Menyimpan nama file untuk digunakan kembali nantinya.
            st.session_state.uploaded_file_name = uploaded_file.name


    # Tampilkan tombol "Bersihkan" jika ada file yang di-upload
    if 'uploaded_file' in st.session_state:
        # Cek apakah preprocessing sudah dilakukan
        if 'preprocessed' not in st.session_state or not st.session_state.preprocessed:
            # Jika tombol "Bersihkan" ditekan, lakukan preprocessing
            if st.button("Bersihkan"):
                df_preprocessed = preprocess_data(st.session_state.uploaded_file)
                
                # Simpan hasil preprocessing di session_state
                if 'preprocessed_data' not in st.session_state:
                    st.session_state.preprocessed_data = []  # Inisialisasi list jika belum ada
                # Simpan data yang diproses dan nama file-nya
                st.session_state.preprocessed_data.append({
                    'data': df_preprocessed,
                    'filename': st.session_state.uploaded_file.name
                })  
                st.success("Preprocessing Selesai.")
        else:
            # Warning jika file sudah diproses sebelumnya
            st.warning("File ini sudah di upload")

    # Menampilkan semua data yang telah diproses
    if 'preprocessed_data' in st.session_state:
        for item in st.session_state.preprocessed_data:
            df = item['data'] # Ambil DataFrame yang telah diproses
            filename = item['filename'] # Ambil nama file
            st.write(f"Hasil Preprosesing dari file: {filename}:") # Tampilkan nama file
            st.dataframe(df) # Tampilkan DataFrame



elif page == "Klastering":
    st.header("Analisis Faktor")  # Menampilkan judul halaman

    # Load data hasil preprocessing dari file
    def load_preprocessed_data():
        file_path = 'preprosesing/hasil.txt'  # Ganti dengan path yang sesuai
        return pd.read_csv(file_path)

    df_selected = load_preprocessed_data()

    # Mendefinisikan kalimat untuk setiap centroid
    centroid_sentences = {
        'kompensasi': "kompensasi naik gaji uang pendapatan dapat penghasilan hasil intensif gaji sedikit gaji banyak bonus",
        'kepuasan_kerja': "kepuasan puas kerja karir bahagia sedih dedikasi nyaman lembur jam kerja waktu istirahat cape capek lelah stres stress",
        'aktualisasi': "aktualisasi aktual pengembangan kembang potensi diri kreatif prestasi jabatan jabat gelar industri karir",
        'hubungan_kerja': "hubungan rekan kerja suasana dukungan dukung kolaborasi tempat toksik toxic jahat benci suka teman kawan toleransi takut buruk"
    }

    # Menghitung posisi dalam DataFrame untuk setiap centroid
    num_rows = len(df_selected)
    centroid_positions = {
        int(num_rows * 0.25): centroid_sentences['kompensasi'],
        int(num_rows * 0.50): centroid_sentences['kepuasan_kerja'],
        int(num_rows * 0.75): centroid_sentences['aktualisasi'],
        int(num_rows * 0.90): centroid_sentences['hubungan_kerja']
    }

    # Menyisipkan kalimat ke dalam DataFrame pada posisi yang ditentukan
    for pos, sentence in centroid_positions.items():
        df_selected.at[pos, 'filtered'] = sentence

    # Memastikan semua entri di 'filtered' adalah string untuk proses TF-IDF
    df_selected['filtered'] = df_selected['filtered'].apply(lambda x: str(x))

    texts = df_selected['filtered'].astype(str)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    # Mengambil centroid awal berdasarkan posisi kalimat yang disisipkan
    initial_centroids = X[list(centroid_positions.keys())].toarray()

    if st.button("Klaster"):
        kmeans = KMeans(n_clusters=4, init=initial_centroids, n_init=10, random_state=0)
        kmeans.fit(X)
        df_selected['cluster'] = kmeans.labels_
        
        # Menghitung Silhouette Score
        silhouette_avg = silhouette_score(X, kmeans.labels_)
        # db_score = davies_bouldin_score(X.toarray(), kmeans.labels_)
        st.write(f"**Silhouette Score:** {silhouette_avg:.2f}")
        # st.write(f"**Davies-Bouldin Index:** {db_score:.2f}")

        # Memisahkan klaster menjadi DataFrame yang berbeda
        cluster_0 = df_selected[df_selected['cluster'] == 0][['filtered']].reset_index(drop=True)
        cluster_1 = df_selected[df_selected['cluster'] == 1][['filtered']].reset_index(drop=True)
        cluster_2 = df_selected[df_selected['cluster'] == 2][['filtered']].reset_index(drop=True)
        cluster_3 = df_selected[df_selected['cluster'] == 3][['filtered']].reset_index(drop=True)
        
        # Mendefinisikan label kustom untuk setiap klaster
        label_klaster = ['kompensasi', 'kepuasan kerja', 'aktualisasi', 'hubungan kerja']
    
        # Daftar kalimat yang ingin dihapus
        kalimat_yang_ingin_dihapus = [ 
            "kompensasi naik gaji uang pendapatan dapat penghasilan hasil intensif gaji sedikit gaji banyak bonus",
            "kepuasan puas kerja karir bahagia sedih dedikasi nyaman lembur jam kerja waktu cape capek lelah stres stress",
            "aktualisasi aktual pengembangan kembang potensi diri kreatif prestasi jabatan jabat gelar industri karir",
            "hubungan hubung rekan kerja suasana dukungan dukung kolaborasi tempat toxic toksik jahat benci suka teman kawan toleransi takut senioritas senior trauma"
        ]

        # Fungsi untuk menghapus tanda baca
        def hapus_tandabaca(text):
            if isinstance(text, str):
                # Menghapus tanda baca dari teks
                text = re.sub(r'[^\w\s]', '', text)
            elif isinstance(text, list):
                # Jika teks berupa list, hapus tanda baca untuk setiap kata dalam list
                text = [re.sub(r'[^\w\s]', '', word) for word in text]
            return text

        # Fungsi untuk menghapus kalimat tertentu
        def hapus_kalimat(text):
            if isinstance(text, str):
                # Menghapus kalimat yang ada dalam daftar kalimat_yang_ingin_dihapus
                for kalimat in kalimat_yang_ingin_dihapus:
                    text = text.replace(kalimat, '')  # Menghapus kalimat yang ditemukan
            elif isinstance(text, list):
                # Jika teks berupa list, hapus kalimat tertentu dalam daftar
                text = [word for word in text if word not in kalimat_yang_ingin_dihapus]
            return text

        # Membersihkan data setiap cluster
        cleaned_data_0 = cluster_0.applymap(hapus_tandabaca).applymap(hapus_kalimat)
        cleaned_data_0 = cleaned_data_0.replace('', np.nan).dropna()

        cleaned_data_1 = cluster_1.applymap(hapus_tandabaca).applymap(hapus_kalimat).dropna()
        cleaned_data_1 = cleaned_data_1.replace('', np.nan).dropna()

        cleaned_data_2 = cluster_2.applymap(hapus_tandabaca).applymap(hapus_kalimat).dropna()
        cleaned_data_2 = cleaned_data_2.replace('', np.nan).dropna()
        
        cleaned_data_3 = cluster_3.applymap(hapus_tandabaca).applymap(hapus_kalimat).dropna()
        cleaned_data_3 = cleaned_data_3.replace('', np.nan).dropna()
        
        for i, (label, dataframe_klaster) in enumerate(zip(label_klaster, [cleaned_data_0, cleaned_data_1, cleaned_data_2, cleaned_data_3])):
            st.write(f"### Faktor {label.capitalize()}")
            st.dataframe(dataframe_klaster[['filtered']])

        
        # Menyimpan data yang telah dibersihkan ke file
        cleaned_data_0.to_csv('klaster/kompensasi.txt', sep='\t', index=False, header=True)
        cleaned_data_1.to_csv('klaster/kepuasan kerja.txt', sep='\t', index=False, header=True)
        cleaned_data_2.to_csv('klaster/aktualisasi.txt', sep='\t', index=False, header=True)
        cleaned_data_3.to_csv('klaster/hubungan kerja.txt', sep='\t', index=False, header=True)

elif page == "Analisis Sentimen":
    st.header("Analisis Sentimen Faktor")

    # Load leksikon positif dan negatif
    pos_lexicon = load_lexicon('leksikon/leksikon-pos.json')
    neg_lexicon = load_lexicon('leksikon/leksikon-neg.json')

    # Tombol untuk memulai analisis sentimen
    if st.button("Lakukan proses analisis sentimen"):

        # Load data kluster dari folder 'klaster'
        def load_cluster_data(cluster_name):
            file_path = f'klaster/{cluster_name}.txt'
            if os.path.exists(file_path):
                return pd.read_csv(file_path, sep='\t')
            else:
                st.error(f"File {cluster_name}.txt tidak ditemukan di folder 'klaster'.")
                return pd.DataFrame()  # Kembalikan DataFrame kosong jika file tidak ditemukan

        # Fungsi untuk menghitung sentimen berdasarkan leksikon
        def hitung_sentimen_berdasarkan_leksikon(text):
            if not isinstance(text, str):
                text = ""
            pos_count = sum(1 for word in text.split() if word in pos_lexicon)
            neg_count = sum(1 for word in text.split() if word in neg_lexicon)
            if pos_count > neg_count:
                return 'Positif', 1  # Sentimen positif dan skor
            elif neg_count > pos_count:
                return 'Negatif', -1  # Sentimen negatif dan skor
            else:
                return 'Netral', 0  # Sentimen netral dan skor

        # Label untuk setiap kluster
        label_klaster = ['kompensasi', 'kepuasan kerja', 'aktualisasi', 'hubungan kerja']

        # Menyimpan hasil analisis sentimen dari setiap kluster
        kumpulan_sentimen = {}

        # Memuat dan memproses setiap kluster
        for label in label_klaster:
            dataframe_klaster = load_cluster_data(label)  # Ambil DataFrame dari file .txt
            if not dataframe_klaster.empty:
                # Analisis sentimen dan tambahkan label dan skor
                dataframe_klaster[['sentiment_label', 'sentiment_score']] = dataframe_klaster['filtered'].apply(
                    lambda x: pd.Series(hitung_sentimen_berdasarkan_leksikon(x))
                )
                
                # Simpan DataFrame untuk kluster saat ini
                kumpulan_sentimen[label] = dataframe_klaster

                # Tampilkan hasil analisis sentimen untuk kluster saat ini
                st.write(f"### Analisis Sentimen Faktor {label.capitalize()}")
                st.write(dataframe_klaster[['filtered', 'sentiment_label', 'sentiment_score']])

                # Simpan hasil analisis ke file
                output_file_path = f'analisis/{label}.txt'
                dataframe_klaster[['filtered', 'sentiment_label', 'sentiment_score']].to_csv(
                    output_file_path, sep='\t', index=False, header=['analisis', 'sentiment_label', 'sentiment_score']
                )

elif page == "Visualisasi Data":
    st.header("Visualisasi Data")
    if st.button("Visualisasikan"):
        # Load hasil analisis sentimen dari file
        def memuat_data_sentimen(cluster_name):
            return pd.read_csv(f'analisis/{cluster_name}.txt', sep='\t')

        # Label klaster
        label_klaster = ['kompensasi', 'kepuasan kerja', 'aktualisasi', 'hubungan kerja']

        # Load dan visualisasikan data untuk setiap klaster
        for label in label_klaster:
            dataframe_klaster = memuat_data_sentimen(label)  # Ambil DataFrame dari file
            if not dataframe_klaster.empty:
                jumlah_sentimen = dataframe_klaster['sentiment_label'].value_counts()

                # Buat Pie Chart
                st.subheader(f"Visualisasi Analisis Sentimen Faktor {label.capitalize()}")
                st.write(f"Total data pada faktor {label.capitalize()} sebanyak : {len(dataframe_klaster)}")
                fig, ax = plt.subplots()
                colors = ['#ADD8E6', '#87CEFA', '#4682B4']
                ax.pie(jumlah_sentimen, labels=jumlah_sentimen.index, autopct='%1.1f%%', startangle=90, colors=colors)
                ax.axis('equal')  # Membuat pie chart berbentuk lingkaran.
                st.pyplot(fig)

                # Deskripsi singkat hasil analisis
                st.write(f"Faktor {label.capitalize()}, analisis sentimen menunjukkan distribusi sebagai berikut:")
                for sentiment, count in jumlah_sentimen.items():
                    st.write(f"- **{sentiment}**: {count} ulasan")

