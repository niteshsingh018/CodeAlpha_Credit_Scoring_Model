import pandas as pd
import os

from sklearn.model_selection import train_test_split


# ==========================================
# PHASE 6: TRAIN / TEST SPLIT
# ==========================================

print("\n========================================")
print("         PHASE 6: TRAIN / TEST SPLIT")
print("========================================")


# ==========================================
# FILE PATHS
# ==========================================

x_file = "data/X.csv"
y_file = "data/y.csv"


# ==========================================
# CHECK FILES
# ==========================================

if not os.path.exists(x_file):

    print("\nERROR: X.csv not found!")
    print("Please complete Phase 5 first.")
    print("python src\\prepare_xy.py")
    exit()


if not os.path.exists(y_file):

    print("\nERROR: y.csv not found!")
    print("Please complete Phase 5 first.")
    print("python src\\prepare_xy.py")
    exit()


# ==========================================
# LOAD X AND y
# ==========================================

print("\nLoading X and y...")

X = pd.read_csv(x_file)
y = pd.read_csv(y_file).squeeze("columns")

print("X and y loaded successfully!")


# ==========================================
# DISPLAY ORIGINAL SIZE
# ==========================================

print("\n----------------------------------------")
print("1. ORIGINAL DATA")
print("----------------------------------------")

print("X shape:", X.shape)
print("y shape:", y.shape)


# ==========================================
# CHECK RECORD COUNT
# ==========================================

if len(X) != len(y):

    print("\nERROR: X and y have different sizes!")

    print("X records:", len(X))
    print("y records:", len(y))

    exit()


print("\nX and y have matching records.")


# ==========================================
# TARGET DISTRIBUTION
# ==========================================

print("\n----------------------------------------")
print("2. TARGET DISTRIBUTION")
print("----------------------------------------")

print(
    y.value_counts()
    .sort_index()
    .to_string()
)


print("\nTarget percentage:")

print(
    y.value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
    .to_string()
)


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

print("\n----------------------------------------")
print("3. CREATING TRAIN / TEST DATA")
print("----------------------------------------")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# DISPLAY SPLIT SIZE
# ==========================================

print("\nTraining and testing datasets created!")

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ==========================================
# CHECK TRAIN TARGET
# ==========================================

print("\n----------------------------------------")
print("4. TRAINING TARGET DISTRIBUTION")
print("----------------------------------------")

print(
    y_train.value_counts()
    .sort_index()
    .to_string()
)


print("\nTraining target percentage:")

print(
    y_train.value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
    .to_string()
)


# ==========================================
# CHECK TEST TARGET
# ==========================================

print("\n----------------------------------------")
print("5. TESTING TARGET DISTRIBUTION")
print("----------------------------------------")

print(
    y_test.value_counts()
    .sort_index()
    .to_string()
)


print("\nTesting target percentage:")

print(
    y_test.value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
    .to_string()
)


# ==========================================
# CREATE DATA FOLDER
# ==========================================

os.makedirs("data", exist_ok=True)


# ==========================================
# SAVE TRAINING DATA
# ==========================================

print("\n----------------------------------------")
print("6. SAVING TRAINING DATA")
print("----------------------------------------")

X_train.to_csv(
    "data/X_train.csv",
    index=False
)

y_train.to_csv(
    "data/y_train.csv",
    index=False
)

print("Saved:")
print("data/X_train.csv")
print("data/y_train.csv")


# ==========================================
# SAVE TESTING DATA
# ==========================================

print("\n----------------------------------------")
print("7. SAVING TESTING DATA")
print("----------------------------------------")

X_test.to_csv(
    "data/X_test.csv",
    index=False
)

y_test.to_csv(
    "data/y_test.csv",
    index=False
)

print("Saved:")
print("data/X_test.csv")
print("data/y_test.csv")


# ==========================================
# FINAL CHECK
# ==========================================

print("\n----------------------------------------")
print("8. FINAL CHECK")
print("----------------------------------------")

print("Total records :", len(X))
print("Training      :", len(X_train))
print("Testing       :", len(X_test))

print(
    "Training percentage:",
    round(len(X_train) / len(X) * 100, 2),
    "%"
)

print(
    "Testing percentage:",
    round(len(X_test) / len(X) * 100, 2),
    "%"
)


# ==========================================
# VERIFY
# ==========================================

if len(X_train) + len(X_test) == len(X):

    print("\nTrain/test split verified successfully.")

else:

    print("\nERROR: Train/test record count mismatch!")


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n========================================")
print("      PHASE 6 COMPLETED SUCCESSFULLY")
print("========================================")

print("\nCreated files:")

print("1. data/X_train.csv")
print("2. data/X_test.csv")
print("3. data/y_train.csv")
print("4. data/y_test.csv")

print("\nNext phase:")
print("Phase 7 - Logistic Regression")