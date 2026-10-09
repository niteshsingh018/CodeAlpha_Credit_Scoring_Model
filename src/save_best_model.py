import os
import shutil
import pandas as pd


# ==========================================
# PHASE 12: SAVE BEST MODEL
# ==========================================

print("\n" + "=" * 60)
print("              PHASE 12: SAVE BEST MODEL")
print("=" * 60)


# ==========================================
# 1. FILE PATHS
# ==========================================

comparison_file = "results/model_comparison.csv"

best_model_file = "models/best_credit_model.pkl"


model_files = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "Random Forest": "models/random_forest.pkl"
}


# ==========================================
# 2. CHECK COMPARISON FILE
# ==========================================

if not os.path.exists(comparison_file):

    print("\nERROR: Model comparison file not found!")

    print("\nPlease run:")
    print("python src\\model_comparison.py")

    exit()


# ==========================================
# 3. LOAD COMPARISON RESULTS
# ==========================================

print("\nLoading model comparison results...")

results = pd.read_csv(comparison_file)

print("Comparison results loaded successfully!")


# ==========================================
# 4. DISPLAY RESULTS
# ==========================================

print("\nModel comparison:")

print(
    results.to_string(index=False)
)


# ==========================================
# 5. CHECK REQUIRED COLUMNS
# ==========================================

if "Model" not in results.columns:

    print("\nERROR: 'Model' column not found!")

    print("Available columns:")
    print(list(results.columns))

    exit()


if "Overall_Score" not in results.columns:

    print("\nERROR: 'Overall_Score' column not found!")

    print("\nPlease run Phase 11 again.")

    exit()


# ==========================================
# 6. FIND BEST MODEL
# ==========================================

best_index = results["Overall_Score"].idxmax()

best_model_name = results.loc[
    best_index,
    "Model"
]

best_score = results.loc[
    best_index,
    "Overall_Score"
]


print("\n" + "=" * 60)
print("                BEST MODEL")
print("=" * 60)

print(
    f"\nModel         : {best_model_name}"
)

print(
    f"Overall Score : {best_score:.4f}"
)


# ==========================================
# 7. GET MODEL FILE
# ==========================================

if best_model_name not in model_files:

    print("\nERROR: Model file mapping not found!")

    print(
        f"Best model detected: {best_model_name}"
    )

    print("\nAvailable model mappings:")

    for name in model_files:
        print(name)

    exit()


source_model_file = model_files[
    best_model_name
]


# ==========================================
# 8. CHECK MODEL FILE
# ==========================================

if not os.path.exists(source_model_file):

    print("\nERROR: Trained model file not found!")

    print(
        f"Expected file: {source_model_file}"
    )

    print("\nPlease run the corresponding model training phase.")

    exit()


# ==========================================
# 9. CREATE MODELS FOLDER
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)


# ==========================================
# 10. COPY BEST MODEL
# ==========================================

print("\nSaving best model...")

shutil.copy2(
    source_model_file,
    best_model_file
)


# ==========================================
# 11. VERIFY FILE
# ==========================================

if os.path.exists(best_model_file):

    file_size = os.path.getsize(
        best_model_file
    )

    print("\nBest model saved successfully!")

    print(
        f"Source model : {source_model_file}"
    )

    print(
        f"Final model  : {best_model_file}"
    )

    print(
        f"File size    : {file_size} bytes"
    )

else:

    print("\nERROR: Best model was not saved!")

    exit()


# ==========================================
# 12. SAVE MODEL INFORMATION
# ==========================================

model_info = pd.DataFrame({
    "Best_Model": [best_model_name],
    "Overall_Score": [best_score]
})


if "Accuracy" in results.columns:
    model_info["Accuracy"] = [
        results.loc[best_index, "Accuracy"]
    ]

if "Precision" in results.columns:
    model_info["Precision"] = [
        results.loc[best_index, "Precision"]
    ]

if "Recall" in results.columns:
    model_info["Recall"] = [
        results.loc[best_index, "Recall"]
    ]

if "F1-Score" in results.columns:
    model_info["F1-Score"] = [
        results.loc[best_index, "F1-Score"]
    ]

if "ROC-AUC" in results.columns:
    model_info["ROC-AUC"] = [
        results.loc[best_index, "ROC-AUC"]
    ]


model_info_file = "results/best_model_info.csv"

model_info.to_csv(
    model_info_file,
    index=False
)


# ==========================================
# 13. FINAL OUTPUT
# ==========================================

print("\n" + "=" * 60)
print("       PHASE 12 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nFinal files:")

print("1. models/best_credit_model.pkl")
print("2. results/best_model_info.csv")

print("\nSelected model:")
print(best_model_name)

print("\nNext Phase:")
print("PHASE 13 - PREDICTION SYSTEM")