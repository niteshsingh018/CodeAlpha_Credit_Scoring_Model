import pandas as pd
import os


# ==========================================
# PHASE 5: PREPARE X AND y
# ==========================================

print("\n========================================")
print("          PHASE 5: PREPARE X AND y")
print("========================================")


# ==========================================
# FILE PATHS
# ==========================================

input_file = "data/credit_risk_features.csv"

x_output_file = "data/X.csv"
y_output_file = "data/y.csv"


# ==========================================
# CHECK INPUT FILE
# ==========================================

if not os.path.exists(input_file):

    print("\nERROR: Feature dataset not found!")

    print("Expected file:")
    print(input_file)

    print("\nPlease complete Phase 4 first:")
    print("python src\\feature_engineering.py")

    exit()


# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading feature dataset...")

df = pd.read_csv(input_file)

print("Feature dataset loaded successfully!")

print("\nDataset shape:")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ==========================================
# CHECK TARGET COLUMN
# ==========================================

print("\n----------------------------------------")
print("1. CHECKING TARGET COLUMN")
print("----------------------------------------")

target_column = "loan_status"

if target_column not in df.columns:

    print("\nERROR: loan_status column not found!")

    print("\nAvailable columns:")

    for column in df.columns:
        print("-", column)

    exit()


print("Target column found:", target_column)


# ==========================================
# CREATE X AND y
# ==========================================

print("\n----------------------------------------")
print("2. CREATING X AND y")
print("----------------------------------------")


# y = target
y = df[target_column].copy()


# X = all columns except target
X = df.drop(
    columns=[target_column]
).copy()


print("\nX created successfully.")
print("y created successfully.")


# ==========================================
# CHECK X DATA TYPES
# ==========================================

print("\n----------------------------------------")
print("3. CHECKING X DATA TYPES")
print("----------------------------------------")


non_numeric_columns = X.select_dtypes(
    exclude=["int64", "float64"]
).columns


if len(non_numeric_columns) > 0:

    print("\nNon-numeric columns found:")

    for column in non_numeric_columns:
        print("-", column)

    print("\nConverting non-numeric columns...")

    X = pd.get_dummies(
        X,
        columns=list(non_numeric_columns),
        drop_first=True,
        dtype=int
    )

else:

    print("All X features are numeric.")


# ==========================================
# HANDLE MISSING VALUES IN X
# ==========================================

print("\n----------------------------------------")
print("4. CHECKING MISSING VALUES")
print("----------------------------------------")


missing_x = X.isnull().sum().sum()
missing_y = y.isnull().sum()


print("Missing values in X:", missing_x)
print("Missing values in y:", missing_y)


if missing_x > 0:

    print("\nFilling missing values in X...")

    for column in X.columns:

        if X[column].isnull().sum() > 0:

            if pd.api.types.is_numeric_dtype(X[column]):

                X[column] = X[column].fillna(
                    X[column].median()
                )

            else:

                X[column] = X[column].fillna(
                    X[column].mode()[0]
                )


if missing_y > 0:

    print("\nRemoving rows with missing target values...")

    valid_rows = y.notna()

    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)


# ==========================================
# CONVERT TARGET TO NUMERIC
# ==========================================

print("\n----------------------------------------")
print("5. PREPARING TARGET VARIABLE")
print("----------------------------------------")


print("Original target values:")

print(
    y.value_counts().to_string()
)


# Convert target if necessary
if not pd.api.types.is_numeric_dtype(y):

    print("\nTarget is categorical.")

    unique_values = y.dropna().unique()

    print("Unique target values:")

    for value in unique_values:
        print("-", value)


    # Convert binary target to 0 and 1
    if len(unique_values) == 2:

        target_mapping = {
            unique_values[0]: 0,
            unique_values[1]: 1
        }

        y = y.map(target_mapping)

        print("\nTarget mapping:")

        for key, value in target_mapping.items():
            print(f"{key} -> {value}")

    else:

        print(
            "\nERROR: Target contains more than two classes."
        )

        exit()


# ==========================================
# FINAL TARGET CHECK
# ==========================================

if y.isnull().sum() > 0:

    print("\nERROR: Target contains invalid values.")

    exit()


print("\nFinal target values:")

print(
    y.value_counts().sort_index().to_string()
)


# ==========================================
# CONVERT X TO NUMERIC
# ==========================================

print("\n----------------------------------------")
print("6. FINAL FEATURE CHECK")
print("----------------------------------------")


for column in X.columns:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )


# Handle values created during conversion
X = X.fillna(0)


print("All X features are numeric.")


# ==========================================
# FINAL SHAPES
# ==========================================

print("\n----------------------------------------")
print("7. X AND y SHAPES")
print("----------------------------------------")

print("X shape:", X.shape)
print("y shape:", y.shape)


# ==========================================
# DISPLAY FEATURES
# ==========================================

print("\n----------------------------------------")
print("8. FEATURES USED FOR MACHINE LEARNING")
print("----------------------------------------")

for i, column in enumerate(X.columns, 1):

    print(f"{i}. {column}")


# ==========================================
# SAVE X
# ==========================================

print("\n----------------------------------------")
print("9. SAVING X")
print("----------------------------------------")

os.makedirs("data", exist_ok=True)

X.to_csv(
    x_output_file,
    index=False
)

print("X saved successfully:")
print(x_output_file)


# ==========================================
# SAVE y
# ==========================================

print("\n----------------------------------------")
print("10. SAVING y")
print("----------------------------------------")

y.to_csv(
    y_output_file,
    index=False,
    header=True
)

print("y saved successfully:")
print(y_output_file)


# ==========================================
# FINAL CHECK
# ==========================================

print("\n----------------------------------------")
print("11. FINAL CHECK")
print("----------------------------------------")

print("X rows:", len(X))
print("y rows:", len(y))

if len(X) == len(y):

    print("X and y have the same number of records.")

else:

    print("ERROR: X and y sizes do not match!")


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n========================================")
print("       PHASE 5 COMPLETED SUCCESSFULLY")
print("========================================")

print("\nCreated files:")

print("1. data/X.csv")
print("2. data/y.csv")

print("\nNext phase:")
print("Phase 6 - Train/Test Split")