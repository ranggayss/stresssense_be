import requests
import json

# URL di mana API Anda berjalan
url = 'http://127.0.0.1:5000/predict'

# Contoh data input (harus berisi semua fitur yang diharapkan)
# Ganti nilai-nilai ini untuk mencoba skenario yang berbeda
# sample_data = {
#     'anxiety_level': 1, 'self_esteem': 28, 'mental_health_history': 0, 'depression': 2,
#     'headache': 0, 'blood_pressure': 2, 'sleep_quality': 5, 'breathing_problem': 0,
#     'noise_level': 1, 'living_conditions': 5, 'safety': 5, 'basic_needs': 5,
#     'academic_performance': 5, 'study_load': 1, 'teacher_student_relationship': 5,
#     'future_career_concerns': 0, 'social_support': 3, 'peer_pressure': 0,
#     'extracurricular_activities': 5, 'bullying': 0
# }

sample_data = {
    'anxiety_level': 20, 'self_esteem': 1, 'mental_health_history': 1, 'depression': 20,
    'headache': 20, 'blood_pressure': 3, 'sleep_quality': 0, 'breathing_problem': 1,
    'noise_level': 5, 'living_conditions': 1, 'safety': 1, 'basic_needs': 1,
    'academic_performance': 1, 'study_load': 5, 'teacher_student_relationship': 1,
    'future_career_concerns': 1, 'social_support': 1, 'peer_pressure': 1,
    'extracurricular_activities': 1, 'bullying': 1
}

# Kirim request POST dengan data JSON
response = requests.post(url, json=sample_data)

# Cetak status code dan hasil response
print(f"Status Code: {response.status_code}")
print(f"Response JSON: {response.json()}")