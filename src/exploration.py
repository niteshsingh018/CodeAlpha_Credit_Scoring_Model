import pandas as pd
import os


# ==========================================
# PHASE 2: DATA EXPLORATION
# ==========================================

print("\n========================================")
print("          PHASE 2: DATA EXPLORATION")
print("========================================")


# ==========================================
# LOAD DATASET
# ==========================================

dataset_path = "data/Loan_approval_data_2025.csv"

if not os.path.exists(dataset_path):
    print("\nERROR: Dataset not found!")
    print("Expected:")
    print(dataset_path)
    exit()


df = pd.read_csv(dataset_path)

print("\nDataset loaded successfully!")


# ==========================================
# BASIC INFORMATION
# ==========================================

print("\n----------------------------------------")
print("1. BASIC DATASET INFORMATION")
print("----------------------------------------")

print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ==========================================
# COLUMN INFORMATION
# ==========================================

print("\n----------------------------------------")
print("2. COLUMN INFORMATION")
print("----------------------------------------")

print(df.info())


# ==========================================
# COLUMN NAMES
# ==========================================

print("\n----------------------------------------")
print("3. COLUMN NAMES")
print("----------------------------------------")

for i, column in enumerate(df.columns, 1):
    print(f"{i}. {column}")


# ==========================================
# MISSING VALUES
# ==========================================

print("\n----------------------------------------")
print("4. MISSING VALUES")
print("----------------------------------------")

missing = df.isnull().sum()

missing_table = pd.DataFrame({
    "Column": missing.index,
    "Missing_Values": missing.values
})

print(missing_table.to_string(index=False))


# ==========================================
# DUPLICATE VALUES
# ==========================================

print("\n----------------------------------------")
print("5. DUPLICATE RECORDS")
print("----------------------------------------")

duplicates = df.duplicated().sum()

print("Duplicate records:", duplicates)


# ==========================================
# NUMERICAL COLUMNS
# ==========================================

print("\n----------------------------------------")
print("6. NUMERICAL COLUMNS")
print("----------------------------------------")

numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

print("Numerical columns:")

for column in numerical_columns:
    print("-", column)


# ==========================================
# CATEGORICAL COLUMNS
# ==========================================

print("\n----------------------------------------")
print("7. CATEGORICAL COLUMNS")
print("----------------------------------------")

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

print("Categorical columns:")

for column in categorical_columns:
    print("-", column)


# ==========================================
# STATISTICAL SUMMARY
# ==========================================

print("\n----------------------------------------")
print("8. STATISTICAL SUMMARY")
print("----------------------------------------")

print(df.describe().transpose().to_string())


# ==========================================
# CATEGORICAL VALUE COUNTS
# ==========================================

print("\n----------------------------------------")
print("9. CATEGORICAL VALUE COUNTS")
print("----------------------------------------")

for column in categorical_columns:

    print("\nColumn:", column)

    print(df[column].value_counts().head(15).to_string())


# ==========================================
# TARGET VARIABLE
# ==========================================

print("\n----------------------------------------")
print("10. TARGET VARIABLE: loan_status")
print("----------------------------------------")

if "loan_status" in df.columns:

    print("\nLoan status distribution:")

    target_counts = df["loan_status"].value_counts()

    print(target_counts.to_string())

    print("\nLoan status percentage:")

    target_percentage = (
        df["loan_status"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print(target_percentage.to_string())

else:

    print("ERROR: loan_status column not found!")


# ==========================================
# UNIQUE VALUES
# ==========================================

print("\n----------------------------------------")
print("11. UNIQUE VALUES")
print("----------------------------------------")

for column in df.columns:

    print(
        f"{column}: {df[column].nunique()} unique values"
    )


# ==========================================
# CORRELATION
# ==========================================

print("\n----------------------------------------")
print("12. NUMERICAL CORRELATION")
print("----------------------------------------")

if len(numerical_columns) > 1:

    correlation = df[numerical_columns].corr()

    print(correlation.round(2).to_string())

else:

    print("Not enough numerical columns for correlation.")


# ==========================================
# FIRST 10 RECORDS
# ==========================================

print("\n----------------------------------------")
print("13. SAMPLE DATA")
print("----------------------------------------")

print(df.head(10).to_string(index=False))


# ==========================================
# LAST 5 RECORDS
# ==========================================

print("\n----------------------------------------")
print("14. LAST 5 RECORDS")
print("----------------------------------------")

print(df.tail().to_string(index=False))


# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n========================================")
print("       PHASE 2 COMPLETED SUCCESSFULLY")
print("========================================")

print("\nDataset is ready for Phase 3.")
print("Next phase: Data Cleaning")
