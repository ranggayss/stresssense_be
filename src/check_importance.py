import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("Memuat model yang sudah dilatih...")
# Pastikan path ini sesuai dengan lokasi model Anda
model = joblib.load('../models/stress_model_v1.pkl')

# Model tidak menyimpan nama fitur, jadi kita perlu mendefinisikannya lagi
# Salin daftar ini dari file src/api.py Anda agar sama persis
feature_names = [
    'anxiety_level', 'self_esteem', 'mental_health_history', 'depression',
    'headache', 'blood_pressure', 'sleep_quality', 'breathing_problem',
    'noise_level', 'living_conditions', 'safety', 'basic_needs',
    'academic_performance', 'study_load', 'teacher_student_relationship',
    'future_career_concerns', 'social_support', 'peer_pressure',
    'extracurricular_activities', 'bullying'
]

# Ekstrak skor feature importance dari model XGBoost
importances = model.feature_importances_

# Buat DataFrame agar mudah dibaca dan diurutkan
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Cetak tabel feature importance
print("\n--- Tingkat Kepentingan Fitur (dari yang Paling Berpengaruh) ---")
print(feature_importance_df)

# Buat dan simpan visualisasi agar lebih mudah dipahami
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance untuk Prediksi Tingkat Stres')
plt.xlabel('Skor Kepentingan')
plt.ylabel('Fitur')
plt.tight_layout()
plt.savefig('feature_importance.png')

print("\nGrafik feature importance berhasil disimpan sebagai 'feature_importance.png'")