import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import requests
from datetime import datetime, timedelta
import time


def fetch_from_thingspeak_and_save_to_csv():
    try:
        # Specify the end time as the current time
        end_time = datetime.utcnow() 
    
        # Hitung waktu mulai sebagai 15 menit sebelum waktu akhir
        start_time = end_time - timedelta(minutes=15)

        # Konversi waktu ke format yang diperlukan (YYYY-MM-DDTHH:MM:SSZ)
        start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_time_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Ganti dengan ID saluran ThingSpeak dan API key Anda yang sebenarnya
        url = f'https://api.thingspeak.com/channels/2566396/feeds.json?api_key=AQ3SV4JV3Z6341SS&start={start_time_str}&end={end_time_str}'

        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            feeds = data['feeds']
            df = pd.DataFrame(feeds, columns=['field1', 'field2', 'field5'])

            df['field5'] = df['field5'].apply(lambda x: None if x == 2000 else x)
            # Hapus baris duplikat untuk integritas data yang lebih baik
            df.drop_duplicates(inplace=True)

            # Simpan data ke file CSV
            df.to_csv('realtimedata.csv', index=False)
            print("Data baru diambil dari ThingSpeak dan disimpan ke realtimedata.csv")
        else:
            print("Gagal mengambil data dari ThingSpeak")

    except Exception as e:
        print(f"Terjadi kesalahan: {str(e)}")

def send_to_thingspeak(api_key, predicted_class):
    url = f"https://api.thingspeak.com/update?api_key={api_key}&field3={predicted_class}"
    response = requests.post(url)
    if response.status_code == 200:
        print("Data terkirim ke ThingSpeak")
    else:
        print("Gagal mengirim data ke ThingSpeak:", response.status_code)

loop_interval = 1*60
while True:
    fetch_from_thingspeak_and_save_to_csv()

    try:
        data_acuan = pd.read_csv(r'C:\xampp\htdocs\otomasi\DatasetAcuanML.csv', delimiter=',')
        print("Kolom yang ada dalam data_acuan:")
        print(data_acuan.columns)
    except FileNotFoundError:
        print("File CSV 'datamonitoringdanklasifikasi.csv' tidak ditemukan.")
        continue

    data_baru = pd.read_csv('realtimedata.csv')
    

    # Periksa apakah kolom yang diperlukan ada dalam data_acuan
    required_columns = ['field1', 'field2', 'field5']
    if not set(required_columns).issubset(data_acuan.columns):
        print("Kolom yang diperlukan tidak ditemukan dalam data_acuan")
        continue

    # Pisahkan fitur dan target dari data acuan, serta hilangkan nilai NaN
    features = data_acuan[required_columns].dropna()
    target = data_acuan.loc[features.index, "occupancy"]

    # Siapkan fitur baru dan hilangkan nilai NaN
    new_features = data_baru[required_columns].dropna()
    print(new_features)

    # Bagi data menjadi set pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.30, random_state=42)

    # Latih model Random Forest
    model = RandomForestClassifier(max_depth=None, n_estimators=50, min_samples_split=10)
    model.fit(X_train, y_train)

    # Prediksi
    average_features = new_features.mean(axis=0).values.reshape(1, -1)
    print(average_features)
    predicted_class = model.predict(average_features)[0]
    print(f"Prediksi Tingkat Okupansi dengan Random Forest: {predicted_class}")

    # Kirim prediksi
    send_to_thingspeak("5QAH99OE3XMDZL9S", predicted_class)

    # Tunggu sebelum iterasi berikutnya
    print(f"Menunggu selama {loop_interval} detik sebelum iterasi berikutnya...")
    time.sleep(loop_interval)