import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier


# ==========================================
# PHASE 9: RANDOM FOREST
# ==========================================

print("\n" + "=" * 50)
print("          PHASE 9: RANDOM FOREST")
print("=" * 50)


# ==========================================
# 1. FILE PATHS
# ==========================================

X_train_file = "data/X_train.csv"
X_test_file = "data/X_test.csv"
y_train_file = "data/y_train.csv"
y_test_file = "data/y_test.csv"

model_file = "models/random_forest.pkl"
prediction_file = "results/random_forest_predictions.csv"
importance_file = "results/random_forest_feature_importance.csv"


# ==========================================
# 2. CHECK REQUIRED FILES
# ==========================================

required_files = [
    X_train_file,
    X_test_file,
    y_train_file,
    y_test_file
]

for file in required_files:
    if not os.path.exists(file):
        print(f"\nERROR: Required file not found!")
        print(f"Missing file: {file}")
        print("\nPlease complete Phase 6 first.")
        exit()


# ==========================================
# 3. CREATE FOLDERS
# ==========================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ==========================================
# 4. LOAD DATA
# ==========================================

print("\nLoading training and testing data...")

X_train = pd.read_csv(X_train_file)
X_test = pd.read_csv(X_test_file)

y_train = pd.read_csv(y_train_file).squeeze()
y_test = pd.read_csv(y_test_file).squeeze()


print("Data loaded successfully!")

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ==========================================
# 5. CREATE RANDOM FOREST MODEL
# ==========================================

print("\nCreating Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    criterion="gini",
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)


# ==========================================
# 6. TRAIN MODEL
# ==========================================

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Random Forest training completed!")


# ==========================================
# 7. MAKE PREDICTIONS
# ==========================================

print("\nMaking predictions...")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

print("Predictions completed!")


# ==========================================
# 8. SAVE MODEL
# ==========================================

joblib.dump(model, model_file)

print("\nRandom Forest model saved successfully!")
print(f"Model file: {model_file}")


# ==========================================
# 9. SAVE PREDICTIONS
# ==========================================

prediction_results = pd.DataFrame({
    "actual": y_test.values,
    "predicted": y_pred,
    "probability": y_probability
})

prediction_results.to_csv(
    prediction_file,
    index=False
)

print("\nPredictions saved successfully!")
print(f"Prediction file: {prediction_file}")


# ==========================================
# 10. FEATURE IMPORTANCE
# ==========================================

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

feature_importance.to_csv(
    importance_file,
    index=False
)


# ==========================================
# 11. DISPLAY TOP FEATURES
# ==========================================

print("\n" + "=" * 50)
print("       TOP 15 IMPORTANT FEATURES")
print("=" * 50)

print(
    feature_importance.head(15).to_string(index=False)
)


# ==========================================
# 12. FINAL OUTPUT
# ==========================================

print("\n" + "=" * 50)
print("      PHASE 9 COMPLETED SUCCESSFULLY")
print("=" * 50)

print("\nFiles created:")

print(f"1. {model_file}")
print(f"2. {prediction_file}")
print(f"3. {importance_file}")

print("\nNext Phase:")
print("PHASE 10 - MODEL EVALUATION")