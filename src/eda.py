# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_classif, RFECV, RFE
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn import metrics
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# %%
plt.style.use('seaborn-v0_8-darkgrid')
blue_palette = sns.color_palette("Blues_r", n_colors=8)
green_palette = sns.color_palette("Greens_r", n_colors=8)
orange_palette = sns.color_palette("Oranges_r", n_colors=8)
sns.set_palette("viridis")

df = pd.read_csv('../data/StressLevelDataset.csv')
print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Dataset Shape: {df.shape}")
print(f"Total Records: {df.shape[0]}")
print(f"Total Features: {df.shape[1]}")
print("\nColumn Names:")
print(df.columns.tolist())

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# %%
print("\n" + "=" * 80)
print("DATA QUALITY CHECK")
print("=" * 80)
print("\nData Types:")
print(df.dtypes)

print("\n Missing Values:")
missing_values = df.isnull().sum()
if missing_values.sum() == 0:
    print("No missing values found.")
else:
    print(missing_values[missing_values > 0])

duplicates = df.duplicated().sum()
print(f"\nDuplicate Records: {duplicates}")

print("\n Basic Statistics:")
print(df.describe())

# %%
print("\n" + "=" * 80)
print("UNIVARIATE ANALYSIS")
print("=" * 80)

fig, axes = plt.subplots(7, 3, figsize=(20, 28))
axes = axes.ravel()

for idx, col in enumerate(df.columns):
    axes[idx].hist(df[col], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')

    mean_val = df[col].mean()
    median_val = df[col].median()
    axes[idx].axvline(mean_val, color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    axes[idx].axvline(median_val, color='darkorange', linestyle='--', linewidth=2, label=f'Median: {median_val:2f}')
    axes[idx].legend()

plt.tight_layout()
plt.savefig('../data/images/feature_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nSkewness of Features:")
skewness = df.skew()
print(skewness.sort_values(ascending=False))

# %%
print("\n" + "=" * 80)
print("TARGET VARIABLE ANALYSIS (stress_level)")
print("=" * 80)

plt.figure(figsize=(10, 6))
stress_counts = df['stress_level'].value_counts().sort_index()
bars = plt.bar(stress_counts.index, stress_counts.values, edgecolor='black', color='teal', alpha=0.8)
plt.xlabel('Stress Level', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Distribution of Stress Levels', fontsize=14, fontweight='bold')
for i, v in enumerate(stress_counts.values):
    plt.text(stress_counts.index[i], v + 5, str(v), ha='center', va='bottom', fontweight='bold')
plt.savefig('../data/images/stress_level_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nStress Level Value Counts:")
print(stress_counts)
print(f"\nPercentage Distribution:")
print((stress_counts / len(df) * 100).round(2))

# %%
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

correlation_matrix = df.corr()

plt.figure(figsize=(16, 14))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='YlGnBu', center=0, square=True, linewidths=1, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('../data/images/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

stress_correlations = correlation_matrix['stress_level'].sort_values(ascending=False)
print("\nTop 10 Features Correlated with Stress Level:")
print(stress_correlations.head(10))
print("\nBottom 10 Features Correlated with Stress Level:")
print(stress_correlations.tail(10))

plt.figure(figsize=(10, 8))
colors = ['darkgreen' if x > 0 else 'darkorange' for x in stress_correlations]
stress_correlations.plot(kind='barh', color=colors)
plt.xlabel('Correlation with Stress Level', fontsize=12)
plt.title('Feature Correlation with Stress Level', fontsize=14, fontweight='bold')
plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig('../data/images/stress_level_correlations.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
print("\n" + "=" * 80)
print("MULTICOLLINEARITY DETECTION")
print("=" * 80)

high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append({
                'Feature 1': correlation_matrix.columns[i],
                'Feature 2': correlation_matrix.columns[j],
                'Correlation': correlation_matrix.iloc[i, j]
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', ascending=False)
    print("\nHighly Correlated Feature Pairs (|correlation| > 0.7):")
    print(high_corr_df)
else:
    print("\nNo highly correlated feature pairs found (|correlation| > 0.7).")

# %%
print("\n" + "=" * 80)
print("BIVARIATE ANALYSIS")
print("=" * 80)

top_features = stress_correlations.abs().nlargest(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.ravel()

for idx, feature in enumerate(top_features):
    data_to_plot = [df[df['stress_level'] == level][feature].values for level in sorted(df['stress_level'].unique())]
    bp = axes[idx].boxplot(data_to_plot, patch_artist=True)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(bp['boxes'])))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    for element in ['whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color='darkblue')

    axes[idx].set_title(f'{feature} vs Stress Level', fontsize=12)
    axes[idx].set_xlabel('Stress Level')
    axes[idx].set_ylabel(feature)
    axes[idx].set_xticklabels(sorted(df['stress_level'].unique()))

plt.suptitle('Top Features vs Stress Level', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../data/images/feature_vs_stress_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
print("\n" + "=" * 80)
print("OUTLIER DETECTION")
print("=" * 80)

outlier_summary = {}
for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    outlier_summary[col] = {
        'count' : len(outliers),
        'percentage' : (len(outliers) / len(df) * 100)
    }

outlier_df = pd.DataFrame(outlier_summary).T
outlier_df = outlier_df.sort_values('count', ascending=False)
print("\nOutlier Summary (IQR Method):")
print(outlier_df[outlier_df['count'] > 0])

plt.figure(figsize=(12, 6))
outlier_df[outlier_df['count'] > 0]['percentage'].plot(kind='bar', color='seagreen', alpha=0.8)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Percentage of Outliers', fontsize=12)
plt.title('Percentage of Outliers by Feature', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../data/images/outlier_percentages.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

X = df.drop('stress_level', axis=1)
y = df['stress_level']

mi_scores = mutual_info_regression(X, y, random_state=42)
mi_scores_df = pd.DataFrame({
    'Feature': X.columns,
    'MI Score': mi_scores
}).sort_values('MI Score', ascending=False)

print("\nMutual Information Scores:")
print(mi_scores_df)

plt.figure(figsize=(10, 8))
colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(mi_scores_df)))
plt.barh(mi_scores_df['Feature'], mi_scores_df['MI Score'], color=colors)
plt.xlabel('Mutual Information Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Feature Importance based on Mutual Information', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('../data/images/mutual_information_scores.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
print("\n" + "=" * 80)
print("DIMENSIONALITY REDUCTION (PCA)")
print("=" * 80)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'o-', color='steelblue', markersize=8, linewidth=2)
ax1.set_xlabel('Principal Component', fontsize=12)
ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
ax1.set_title('PCA Scree Plot', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'o-', color='darkgreen', markersize=8, linewidth=2)
ax2.axhline(y=0.95, color='darkorange', linestyle='--', linewidth=2, label='95% Variance')
ax2.set_xlabel('Number of Components', fontsize=12)
ax2.set_ylabel('Cumulative Explained Variance', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../data/images/pca_analysis', dpi=300, bbox_inches='tight')
plt.show()

n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"\nNumber of components needed for 95% variance: {n_components_95}")

# %%
print("\n" + "=" * 80)
print("STATISTICAL TESTS")
print("=" * 80)

statistic, p_value = stats.normaltest(df['stress_level'])
print(f"\nNormality Test for Stress Level:")
print(f"Statistic: {statistic:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Is normally distributed? {'Yes' if p_value > 0.05 else 'No'}")

print("\nANOVA Tests (Feature groups by stress level):")
for feature in ['anxiety_level', 'depression', 'academic_performance']:
    groups = [group[feature].values for name, group in df.groupby('stress_level')]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"\n{feature}:")
    print(f" F-statistic: {f_stat:.4f}")
    print(f" P-value: {p_val:.4f}")
    print(f" Significant difference? {'Yes' if p_val < 0.05 else 'No'}")

# %%
print("\n" + "=" * 80)
print("RECOMMENDATIONS FOR ML PREPROCESSING")
print("=" * 80)

print("\n1. FEATURE SCALING:")
print("   - All features are on similar scales (mostly 0-5 range)")
print("   - StandardScaler or MinMaxScaler recommended for algorithms sensitive to scale")
print("   - Tree-based models may not require scaling")

print("\n2. HANDLING MULTICOLLINEARITY:")
if high_corr_pairs:
    print("   - Consider removing one feature from highly correlated pairs")
    print("   - Or use dimensionality reduction techniques (PCA, LDA)")
else:
    print("   - No severe multicollinearity detected")

print("\n3. OUTLIER TREATMENT:")
if outlier_df[outlier_df['count'] > 0].shape[0] > 0:
    print("   - Consider capping outliers or using robust scaling")
    print("   - Tree-based models are generally robust to outliers")
else:
    print("   - Minimal outliers detected")

print("\n4. FEATURE ENGINEERING SUGGESTIONS:")
print("   - Create interaction features between highly correlated variables")
print("   - Consider polynomial features for non-linear relationships")
print("   - Group similar features (e.g., physical symptoms, academic factors)")

print("\n5. CLASS IMBALANCE:")
stress_dist = df['stress_level'].value_counts(normalize=True)
if stress_dist.min() < 0.1:
    print("   - Consider using SMOTE or class weights for imbalanced classes")
else:
    print("   - Classes are reasonably balanced")

print("\n6. FEATURE SELECTION:")
print("   - Use mutual information scores for initial feature selection")
print("   - Consider recursive feature elimination with cross-validation")
print(f"   - Start with top {len(mi_scores_df[mi_scores_df['MI Score'] > 0.1])} features based on MI scores")

# %%
summary_report = f"""
STRESS LEVEL DATASET ANALYSIS SUMMARY
=====================================

Dataset Overview:
- Total Records: {df.shape[0]}
- Total Features: {df.shape[1]}
- No Missing Values: {missing_values.sum() == 0}
- Duplicate Rows: {duplicates}

Target Variable Distribution:
{stress_counts.to_dict()}

Top 5 Features Correlated with Stress Level:
{stress_correlations.head(5).to_dict()}

Feature Importance (Top 5 by Mutual Information):
{mi_scores_df.head(5).to_dict()}

Dimensionality Reduction:
- Components for 95% variance: {n_components_95}

Outliers Detected:
- Features with >5% outliers: {len(outlier_df[outlier_df['percentage'] > 5])}

Preprocessing Recommendations:
1. Scaling: Recommended (StandardScaler/MinMaxScaler)
2. Feature Selection: Start with top {len(mi_scores_df[mi_scores_df['MI Score'] > 0.1])} features
3. Handle multicollinearity if needed
4. Consider ensemble methods for robustness
"""

# Save summary to file
with open('../data/result/analysis_summary.txt', 'w') as f:
    f.write(summary_report)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nFiles saved:")
print("- feature_distributions.png")
print("- stress_level_distribution.png")
print("- correlation_matrix.png")
print("- stress_correlations.png")
print("- feature_vs_stress_boxplots.png")
print("- outlier_percentages.png")
print("- mutual_information_scores.png")
print("- pca_analysis.png")
print("- analysis_summary.txt")
print("\nReady for ML modeling!")

# %%
# ==============================================================================
# BAGIAN 1: PERSIAPAN DATA (TERMASUK SCALING)
# ==============================================================================

# 1. Pembagian Data (Sama seperti sebelumnya)
# Pastikan X dan y sudah didefinisikan sebelumnya dari data Anda
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 2. Scaling Fitur
# Scaling penting untuk beberapa model dan merupakan praktik yang baik.
print("Melakukan scaling pada fitur...")
scaler = StandardScaler()

# Kita "belajar" (fit) scaling HANYA dari data latih untuk menghindari kebocoran data
X_train_scaled = scaler.fit_transform(X_train)

# Kita menerapkan (transform) scaling yang sama ke data uji
X_test_scaled = scaler.transform(X_test)


# ==============================================================================
# BAGIAN 2: HYPERPARAMETER TUNING DENGAN GRIDSEARCHCV
# ==============================================================================
# Kita akan mencari parameter terbaik untuk XGBClassifier

print("Memulai pencarian hyperparameter terbaik untuk XGBClassifier...")
# 3. Inisialisasi model yang akan di-tuning
model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

# 4. Tentukan "kamus" parameter yang ingin dicoba
# Ini adalah beberapa parameter umum XGBoost. Kita akan mencoba beberapa kombinasi.
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.1, 0.05]
}

# 5. Siapkan GridSearchCV
# cv=5 berarti menggunakan 5-fold cross-validation. Ini cara evaluasi yang lebih stabil.
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# 6. Latih GridSearchCV pada data yang sudah di-scaling
# Proses ini akan mencoba semua kombinasi parameter dan mungkin butuh beberapa saat.
grid_search.fit(X_train_scaled, y_train)

# ==============================================================================
# BAGIAN 3: EVALUASI MODEL TERBAIK
# ==============================================================================

# 7. Tampilkan parameter terbaik yang ditemukan
print("\nParameter terbaik yang ditemukan:")
print(grid_search.best_params_)

# 8. Gunakan model terbaik yang sudah dilatih oleh GridSearchCV untuk membuat prediksi
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test_scaled)

# 9. Tampilkan laporan klasifikasi dari model yang sudah dioptimalkan
print("\nClassification Report dari Model Optimal:")
print(classification_report(y_test, predictions))

# %%
import joblib

# Ambil model terbaik dari grid search
final_model = grid_search.best_estimator_

# Ambil scaler yang sudah dilatih
# Pastikan variabel 'scaler' masih ada dari langkah preprocessing
final_scaler = scaler

# Simpan kedua objek ke dalam folder 'models/'
joblib.dump(final_model, '../models/stress_model_v1.pkl')
joblib.dump(final_scaler, '../models/scaler_v1.pkl')

print("Model dan scaler berhasil disimpan!")


