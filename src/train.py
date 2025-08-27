# Import semua library yang dibutuhkan di awal
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# ==============================================================================
# SCRIPT UTAMA UNTUK MELATIH MODEL
# ==============================================================================

if __name__ == '__main__':
    print("--- Memulai Proses Pelatihan dan Optimasi Model ---")

    # 1. Memuat Dataset
    print("\n[Langkah 1/6] Memuat dataset dari 'data/raw/'...")
    try:
        df = pd.read_csv('data/stressLevelDataset.csv')
        X = df.drop('stress_level', axis=1)
        y = df['stress_level']
    except FileNotFoundError:
        print("ERROR: Pastikan file 'stressLevelDataset.csv' ada di folder 'data/raw/'")
        exit()

    # 2. Pembagian Data
    print("[Langkah 2/6] Membagi data menjadi set latih dan uji...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # 3. Scaling Fitur
    print("[Langkah 3/6] Melakukan scaling pada fitur...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Pencarian Hyperparameter dengan GridSearchCV
    print("[Langkah 4/6] Memulai pencarian hyperparameter terbaik untuk XGBClassifier...")
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.05]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    
    print("\nParameter terbaik yang ditemukan:")
    print(grid_search.best_params_)

    # 5. Evaluasi Model Terbaik
    print("\n[Langkah 5/6] Mengevaluasi model terbaik pada data uji...")
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test_scaled)
    
    print("\nClassification Report dari Model Optimal:")
    print(classification_report(y_test, predictions))

    # 6. Menyimpan Model dan Scaler
    print("\n[Langkah 6/6] Menyimpan model dan scaler ke folder 'models/'...")
    joblib.dump(best_model, 'models/stress_model_v1.pkl')
    joblib.dump(scaler, 'models/scaler_v1.pkl')

    print("\n--- Proses Pelatihan Selesai ---")
    print("File 'stress_model_v1.pkl' dan 'scaler_v1.pkl' telah berhasil disimpan.")