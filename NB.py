
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. LOAD + SPLIT DATA
# =========================
df = pd.read_csv('final_crop_rotation_plan.csv')

soil_features = [
    'Elevation_m', 'SoilDepth_m', 'Drainage',
    'Permeability', 'pH', 'OrganicCarbon_pct'
]

# Split dataset
df_train, df_temp = train_test_split(df, test_size=0.30, random_state=42)
df_val, df_test = train_test_split(df_temp, test_size=0.50, random_state=42)

# Feature / Target split
X_train, y_train = df_train[soil_features], df_train['Recommended_Rotation']
X_val, y_val = df_val[soil_features], df_val['Recommended_Rotation']
X_test, y_test = df_test[soil_features], df_test['Recommended_Rotation']

print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

# =========================
# 2. MODEL TRAINING + TUNING
# =========================
smoothing_options = [1e-9, 1e-5, 1e-1, 1.0]

best_accuracy = 0
best_model = None

print("\n--- Validation Phase ---")
for alpha in smoothing_options:
    model = GaussianNB(var_smoothing=alpha)
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)

    print(f"Alpha: {alpha} → Val Accuracy: {val_acc:.2%}")

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        best_model = model

print("\n✅ Best model selected!")

# =========================
# 3. SPATIAL INDEX (FAST LOOKUP)
# =========================
coordinates = df[['Latitude', 'Longitude']].values
spatial_tree = KDTree(coordinates)

# Pre-extract soil features as NumPy (faster access)
soil_data_np = df[soil_features].values

def predict_crop_rotation_by_coordinates(latitude, longitude):
    _, idx = spatial_tree.query([latitude, longitude])

    input_vector = soil_data_np[idx].reshape(1, -1)
    return best_model.predict(input_vector)[0]

# =========================
# 4. PIPELINE TESTING
# =========================
print("\n=== SYSTEM-WIDE PIPELINE TEST ===")

# Vectorized-style loop (faster than iterrows)
test_coords = df_test[['Latitude', 'Longitude']].values

pipeline_test_preds = [
    predict_crop_rotation_by_coordinates(lat, lon)
    for lat, lon in test_coords
]

pipeline_acc = accuracy_score(y_test, pipeline_test_preds)
print(f"Pipeline Accuracy: {pipeline_acc:.2%}")

# =========================
# 5. RANDOM SPOT CHECKS
# =========================
print("\n=== RANDOM SPOT CHECKS ===")

sample_rows = df_test.sample(5, random_state=42)

for i, row in enumerate(sample_rows.itertuples(index=False), 1):
    pred = predict_crop_rotation_by_coordinates(row.Latitude, row.Longitude)

    print(f"\nTest #{i}")
    print(f"Coordinates: ({row.Latitude:.4f}, {row.Longitude:.4f})")
    print(f"True: {row.Recommended_Rotation}")
    print(f"Pred: {pred}")
    print(f"Match: {'✅' if pred == row.Recommended_Rotation else '❌'}")

# =========================
# 6. PERFORMANCE REPORT
# =========================
print("\n=== MODEL REPORT ===")
print(classification_report(y_test, pipeline_test_preds))

labels = np.unique(y_test)
cm = confusion_matrix(y_test, pipeline_test_preds)

cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print("\nConfusion Matrix:\n", cm_df)

# =========================
# 7. CONFUSION MATRIX VISUAL
# =========================
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels
)

plt.title('Naive Bayes Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=35, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig('confusion_matrix_visual.png', dpi=300)
print("✅ Confusion matrix saved!")

# =========================
# 8. USER INPUT TOOL
# =========================
print("\n--- 🌾 CROP ROTATION CALCULATOR 🌾 ---")

try:
    user_lat = float(input("Enter Latitude: "))
    user_lon = float(input("Enter Longitude: "))

    print("\nProcessing...")

    predicted_plan = predict_crop_rotation_by_coordinates(user_lat, user_lon)

    print("\n--------------------------------")
    print(f"Location: ({user_lat}, {user_lon})")
    print(f"Recommended Rotation: {predicted_plan}")
    print("--------------------------------")

except Exception as e:
    print(f"Error: {e}")
