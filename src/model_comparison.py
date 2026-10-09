import os
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# PHASE 11: MODEL COMPARISON
# ==========================================

print("\n" + "=" * 60)
print("              PHASE 11: MODEL COMPARISON")
print("=" * 60)


# ==========================================
# 1. FILE PATH
# ==========================================

evaluation_file = "results/model_evaluation.csv"

comparison_file = "results/model_comparison.csv"

chart_file = "results/model_comparison.png"


# ==========================================
# 2. CHECK EVALUATION FILE
# ==========================================

if not os.path.exists(evaluation_file):

    print("\nERROR: Evaluation file not found!")

    print("\nPlease run:")
    print("python src\\model_evaluation.py")

    exit()


# ==========================================
# 3. LOAD EVALUATION RESULTS
# ==========================================

print("\nLoading evaluation results...")

results = pd.read_csv(evaluation_file)

print("Evaluation results loaded successfully!")


# ==========================================
# 4. CHECK COLUMNS
# ==========================================

required_columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]

for column in required_columns:

    if column not in results.columns:

        print("\nERROR: Required column missing!")
        print(f"Missing column: {column}")

        print("\nAvailable columns:")
        print(list(results.columns))

        exit()


# ==========================================
# 5. DISPLAY ORIGINAL RESULTS
# ==========================================

print("\n" + "=" * 60)
print("              MODEL PERFORMANCE")
print("=" * 60)

print(
    results.to_string(index=False)
)


# ==========================================
# 6. FIND BEST MODEL FOR EACH METRIC
# ==========================================

print("\n" + "=" * 60)
print("           BEST MODEL BY EACH METRIC")
print("=" * 60)


metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]


for metric in metrics:

    best_index = results[metric].idxmax()

    best_model = results.loc[
        best_index,
        "Model"
    ]

    best_value = results.loc[
        best_index,
        metric
    ]

    print(
        f"{metric:<12} : "
        f"{best_model:<20} "
        f"({best_value:.4f})"
    )


# ==========================================
# 7. CALCULATE OVERALL SCORE
# ==========================================

results["Overall_Score"] = (
    results["Accuracy"]
    + results["Precision"]
    + results["Recall"]
    + results["F1-Score"]
    + results["ROC-AUC"]
) / 5


# ==========================================
# 8. SORT MODELS
# ==========================================

comparison = results.sort_values(
    by="Overall_Score",
    ascending=False
).reset_index(drop=True)


# ==========================================
# 9. ADD RANK
# ==========================================

comparison.insert(
    0,
    "Rank",
    range(1, len(comparison) + 1)
)


# ==========================================
# 10. ROUND VALUES
# ==========================================

comparison[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
        "Overall_Score"
    ]
] = comparison[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
        "Overall_Score"
    ]
].round(4)


# ==========================================
# 11. DISPLAY RANKING
# ==========================================

print("\n" + "=" * 60)
print("                MODEL RANKING")
print("=" * 60)

print(
    comparison[
        [
            "Rank",
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC",
            "Overall_Score"
        ]
    ].to_string(index=False)
)


# ==========================================
# 12. FIND BEST MODEL
# ==========================================

best_model = comparison.iloc[0]


print("\n" + "=" * 60)
print("              OVERALL BEST MODEL")
print("=" * 60)

print(
    f"\nBest Model    : {best_model['Model']}"
)

print(
    f"Accuracy      : {best_model['Accuracy']:.4f}"
)

print(
    f"Precision     : {best_model['Precision']:.4f}"
)

print(
    f"Recall        : {best_model['Recall']:.4f}"
)

print(
    f"F1-Score      : {best_model['F1-Score']:.4f}"
)

print(
    f"ROC-AUC       : {best_model['ROC-AUC']:.4f}"
)

print(
    f"Overall Score : {best_model['Overall_Score']:.4f}"
)


# ==========================================
# 13. SAVE COMPARISON
# ==========================================

comparison.to_csv(
    comparison_file,
    index=False
)

print("\nModel comparison saved successfully!")

print(
    f"File: {comparison_file}"
)


# ==========================================
# 14. CREATE COMPARISON CHART
# ==========================================

chart_data = results.set_index("Model")[metrics]

ax = chart_data.plot(
    kind="bar",
    figsize=(12, 7)
)

plt.title(
    "Credit Scoring Model Comparison"
)

plt.xlabel("Models")

plt.ylabel("Performance Score")

plt.ylim(0, 1)

plt.xticks(rotation=0)

plt.legend(
    title="Metrics"
)

plt.tight_layout()


plt.savefig(
    chart_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Comparison chart saved: {chart_file}"
)


# ==========================================
# 15. FINAL OUTPUT
# ==========================================

print("\n" + "=" * 60)
print("       PHASE 11 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nCreated files:")

print("1. results/model_comparison.csv")
print("2. results/model_comparison.png")

print("\nSelected best model:")
print(best_model["Model"])

print("\nNext Phase:")
print("PHASE 12 - SAVE BEST MODEL")