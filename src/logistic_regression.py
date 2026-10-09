import pandas as pd
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ==========================================
# PHASE 7: LOGISTIC REGRESSION
# ==========================================

print("\n========================================")
print("        PHASE 7: LOGISTIC REGRESSION")
print("========================================")


# ==========================================
# FILE PATHS
# ==========================================

x_train_file = "data/X_train.csv"
x_test_file = "data/X_test.csv"
y_train_file = "data/y_train.csv"
y_test_file = "data/y_test.csv"


# ==========================================
# CHECK FILES
# ==========================================

required_files = [
    x_train_file,
    x_test_file,
    y_train_file,
    y_test_file
]

for file in required_files:

    if not os.path.exists(file):

        print("\nERROR: Required file not found!")
        print("Missing:", file)

        print("\nPlease complete Phase 6 first:")
        print("python src\\train_test_split.py")

        exit()


# ==========================================
# LOAD DATA
# ==========================================

print("\nLoading training and testing data...")

X_train = pd.read_csv(x_train_file)
X_test = pd.read_csv(x_test_file)

y_train = pd.read_csv(y_train_file).squeeze("columns")
y_test = pd.read_csv(y_test_file).squeeze("columns")

print("Data loaded successfully!")


# ==========================================
# DISPLAY DATA SIZE
# ==========================================

print("\n----------------------------------------")
print("1. DATA INFORMATION")
print("----------------------------------------")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ==========================================
# CREATE MODEL
# ==========================================

print("\n----------------------------------------")
print("2. CREATING LOGISTIC REGRESSION MODEL")
print("----------------------------------------")


# StandardScaler is important for Logistic Regression
# because the features have different scales.

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "logistic_regression",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])


print("Logistic Regression model created.")


# ==========================================
# TRAIN MODEL
# ==========================================

print("\n----------------------------------------")
print("3. TRAINING MODEL")
print("----------------------------------------")

print("Training Logistic Regression...")

model.fit(
    X_train,
    y_train
)

print("Model training completed successfully!")


# ==========================================
# PREDICTIONS
# ==========================================

print("\n----------------------------------------")
print("4. MAKING PREDICTIONS")
print("----------------------------------------")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

print("Predictions generated successfully!")


# ==========================================
# DISPLAY SAMPLE PREDICTIONS
# ==========================================

print("\n----------------------------------------")
print("5. SAMPLE PREDICTIONS")
print("----------------------------------------")

prediction_table = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": y_pred[:10],
    "Probability": y_probability[:10]
})

print(
    prediction_table.to_string(index=False)
)


# ==========================================
# SAVE PREDICTIONS
# ==========================================

print("\n----------------------------------------")
print("6. SAVING PREDICTIONS")
print("----------------------------------------")

os.makedirs("results", exist_ok=True)

prediction_results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Probability": y_probability
})

prediction_results.to_csv(
    "results/logistic_predictions.csv",
    index=False
)

print(
    "Predictions saved to:"
    "\nresults/logistic_predictions.csv"
)


# ==========================================
# SAVE MODEL
# ==========================================

print("\n----------------------------------------")
print("7. SAVING MODEL")
print("----------------------------------------")

os.makedirs("models", exist_ok=True)

model_file = "models/logistic_regression.pkl"

joblib.dump(
    model,
    model_file
)

print("Model saved successfully!")

print("Model file:")
print(model_file)


# ==========================================
# MODEL INFORMATION
# ==========================================

print("\n----------------------------------------")
print("8. MODEL INFORMATION")
print("----------------------------------------")

print("Algorithm        : Logistic Regression")
print("Training records :", len(X_train))
print("Testing records  :", len(X_test))
print("Number of features:", X_train.shape[1])


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n========================================")
print("     PHASE 7 COMPLETED SUCCESSFULLY")
print("========================================")

print("\nCreated files:")

print("1. models/logistic_regression.pkl")
print("2. results/logistic_predictions.csv")

print("\nNext phase:")
print("Phase 8 - Decision Tree")