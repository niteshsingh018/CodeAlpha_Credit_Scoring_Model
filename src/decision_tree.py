import pandas as pd
import os
import joblib

from sklearn.tree import DecisionTreeClassifier


# ==========================================
# PHASE 8: DECISION TREE
# ==========================================

print("\n========================================")
print("          PHASE 8: DECISION TREE")
print("========================================")


# ==========================================
# FILE PATHS
# ==========================================

x_train_file = "data/X_train.csv"
x_test_file = "data/X_test.csv"
y_train_file = "data/y_train.csv"
y_test_file = "data/y_test.csv"


# ==========================================
# CHECK REQUIRED FILES
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
# DATA INFORMATION
# ==========================================

print("\n----------------------------------------")
print("1. DATA INFORMATION")
print("----------------------------------------")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ==========================================
# CREATE DECISION TREE
# ==========================================

print("\n----------------------------------------")
print("2. CREATING DECISION TREE MODEL")
print("----------------------------------------")


model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)


print("Decision Tree model created.")

print("\nModel parameters:")
print("Criterion         : gini")
print("Maximum depth     : 8")
print("Minimum split     : 10")
print("Minimum leaf      : 5")


# ==========================================
# TRAIN MODEL
# ==========================================

print("\n----------------------------------------")
print("3. TRAINING MODEL")
print("----------------------------------------")

print("Training Decision Tree...")

model.fit(
    X_train,
    y_train
)

print("Model training completed successfully!")


# ==========================================
# MAKE PREDICTIONS
# ==========================================

print("\n----------------------------------------")
print("4. MAKING PREDICTIONS")
print("----------------------------------------")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

print("Predictions generated successfully!")


# ==========================================
# SAMPLE PREDICTIONS
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
    "results/decision_tree_predictions.csv",
    index=False
)

print(
    "Predictions saved to:"
    "\nresults/decision_tree_predictions.csv"
)


# ==========================================
# SAVE MODEL
# ==========================================

print("\n----------------------------------------")
print("7. SAVING MODEL")
print("----------------------------------------")

os.makedirs("models", exist_ok=True)

model_file = "models/decision_tree.pkl"

joblib.dump(
    model,
    model_file
)

print("Model saved successfully!")

print("Model file:")
print(model_file)


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

print("\n----------------------------------------")
print("8. FEATURE IMPORTANCE")
print("----------------------------------------")

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print(
    feature_importance.head(15).to_string(index=False)
)


# ==========================================
# SAVE FEATURE IMPORTANCE
# ==========================================

feature_importance.to_csv(
    "results/decision_tree_feature_importance.csv",
    index=False
)

print(
    "\nFeature importance saved to:"
    "\nresults/decision_tree_feature_importance.csv"
)


# ==========================================
# MODEL INFORMATION
# ==========================================

print("\n----------------------------------------")
print("9. MODEL INFORMATION")
print("----------------------------------------")

print("Algorithm         : Decision Tree")
print("Training records  :", len(X_train))
print("Testing records   :", len(X_test))
print("Number of features:", X_train.shape[1])


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n========================================")
print("     PHASE 8 COMPLETED SUCCESSFULLY")
print("========================================")

print("\nCreated files:")

print("1. models/decision_tree.pkl")
print("2. results/decision_tree_predictions.csv")
print("3. results/decision_tree_feature_importance.csv")

print("\nNext phase:")
print("Phase 9 - Random Forest")