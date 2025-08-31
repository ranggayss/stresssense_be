from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd 

# Inisialisasi aplikasi Flask
app = Flask(__name__)
CORS(app)

# PEMUATAN MODEL & SCALER
try:
    model = joblib.load('models/stress_model_v1.pkl')
    scaler = joblib.load('models/scaler_v1.pkl')
    print("Model dan scaler berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat model atau scaler: {e}")
    model = None
    scaler = None
    
EXPECTED_FEATURES = [
    'anxiety_level', 'self_esteem', 'mental_health_history', 'depression',
    'headache', 'blood_pressure', 'sleep_quality', 'breathing_problem',
    'noise_level', 'living_conditions', 'safety', 'basic_needs',
    'academic_performance', 'study_load', 'teacher_student_relationship',
    'future_career_concerns', 'social_support', 'peer_pressure',
    'extracurricular_activities', 'bullying'
]

@app.route('/predict', methods=['POST'])
def predict():

    if model is None or scaler is None:
        return jsonify({'error': 'Model atau scaler tidak berhasil dimuat. Cek log server.'}), 500

    # 1. Mengambil data JSON dari request yang dikirim oleh frontend
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Data tidak ditemukan dalam request'}), 400

    try:
        # 2. Mengubah data JSON menjadi array dengan urutan yang benar
        data_array = [data[feature] for feature in EXPECTED_FEATURES]

        # 3. Melakukan scaling pada data input
        data_scaled = scaler.transform([data_array])

        # 4. Melakukan prediksi menggunakan model
        prediction = model.predict(data_scaled)

        # 5. Mengambil hasil prediksi (biasanya berupa array, kita ambil elemen pertamanya)
        result = int(prediction[0])

        stress_labels = {0: 'Tingkat Stres Rendah', 1: 'Tingkat Stres Sedang', 2: 'Tingkat Stres Tinggi'}
        result_label = stress_labels.get(result, 'Tidak Diketahui')

        # 6. Mengembalikan hasil prediksi dalam format JSON
        return jsonify({
            'prediction': result,
            'prediction_label': result_label
        })

    except KeyError as e:
        # Error jika ada fitur yang hilang dari data JSON
        return jsonify({'error': f'Fitur yang hilang: {str(e)}'}), 400
    except Exception as e:
        # Error umum lainnya
        return jsonify({'error': f'Terjadi kesalahan saat prediksi: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)