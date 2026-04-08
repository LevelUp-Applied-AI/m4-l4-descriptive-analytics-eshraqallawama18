"""Lab 4 — Descriptive Analytics: Student Performance EDA"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def load_and_profile(filepath):
    df = pd.read_csv(filepath)

    # Create output folder
    os.makedirs("output", exist_ok=True)

    # Missing values
    missing_counts = df.isnull().sum()
    missing_percent = (missing_counts / len(df)) * 100

    with open("output/data_profile.txt", "w") as f:
        f.write(f"Shape: {df.shape}\n\n")

        f.write("Data Types:\n")
        f.write(str(df.dtypes) + "\n\n")

        f.write("Missing Values:\n")
        summary = pd.DataFrame({
            "count": missing_counts,
            "percent": missing_percent
        })
        f.write(str(summary) + "\n\n")

        f.write("Descriptive Statistics:\n")
        f.write(str(df.describe()) + "\n\n")

        f.write("Handling Decisions:\n")
        f.write("- commute_minutes: Filled with median (~9% missing)\n")
        f.write("- study_hours_weekly: Dropped missing rows\n")

    return df


def plot_distributions(df):
    print("Running distribution plots...")
    os.makedirs("output", exist_ok=True)

    # GPA
    sns.histplot(df["gpa"], kde=True)
    plt.title("GPA Distribution (Most between 2.5–3.5)")
    plt.savefig("output/gpa_distribution.png")
    plt.clf()

    # Study hours
    sns.histplot(df["study_hours_weekly"], kde=True)
    plt.title("Study Hours Distribution")
    plt.savefig("output/study_hours_distribution.png")
    plt.clf()

    # Attendance
    sns.histplot(df["attendance_pct"], kde=True)
    plt.title("Attendance Distribution")
    plt.savefig("output/attendance_distribution.png")
    plt.clf()

    # Boxplot
    sns.boxplot(x="department", y="gpa", data=df)
    plt.xticks(rotation=45)
    plt.title("GPA by Department")
    plt.savefig("output/gpa_by_department.png")
    plt.clf()

    # Bar chart
    sns.countplot(x="scholarship", data=df)
    plt.xticks(rotation=45)
    plt.title("Scholarship Distribution")
    plt.savefig("output/scholarship_distribution.png")
    plt.clf()


def plot_correlations(df):
    print("Running correlation plots...")
    os.makedirs("output", exist_ok=True)

    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()

    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("output/correlation_heatmap.png")
    plt.clf()

    # Scatter plots
    sns.scatterplot(x="study_hours_weekly", y="gpa", data=df)
    plt.title("Study Hours vs GPA")
    plt.savefig("output/study_vs_gpa.png")
    plt.clf()

    sns.scatterplot(x="attendance_pct", y="gpa", data=df)
    plt.title("Attendance vs GPA")
    plt.savefig("output/attendance_vs_gpa.png")
    plt.clf()


def run_hypothesis_tests(df):
    results = {}

    # --- T-Test ---
    yes = df[df["has_internship"] == "Yes"]["gpa"]
    no = df[df["has_internship"] == "No"]["gpa"]

    t_stat, p_val = stats.ttest_ind(yes, no)

    print("\nT-Test (Internship vs GPA)")
    print(f"t-statistic: {t_stat}")
    print(f"p-value: {p_val}")

    results["internship_ttest"] = {
        "t_stat": t_stat,
        "p_value": p_val
    }

    # --- ANOVA ---
    groups = [group["gpa"].values for name, group in df.groupby("department")]
    f_stat, p_val_anova = stats.f_oneway(*groups)

    print("\nANOVA (GPA across departments)")
    print(f"F-statistic: {f_stat}")
    print(f"p-value: {p_val_anova}")

    results["dept_anova"] = {
        "f_stat": f_stat,
        "p_value": p_val_anova
    }

    # --- Correlation Test ---
    corr, p_corr = stats.pearsonr(df["study_hours_weekly"], df["gpa"])

    print("\nCorrelation Test (Study Hours vs GPA)")
    print(f"Correlation: {corr}")
    print(f"p-value: {p_corr}")

    results["correlation_test"] = {
        "correlation": corr,
        "p_value": p_corr
    }

    return results

def main():
    os.makedirs("output", exist_ok=True)

    print("STEP 1: Loading data")
    df = load_and_profile("data/student_performance.csv")

    print("STEP 2: Cleaning data")
    df["commute_minutes"].fillna(df["commute_minutes"].median(), inplace=True)
    df = df.dropna(subset=["study_hours_weekly"])

    print("STEP 3: Plot distributions")
    plot_distributions(df)

    print("STEP 4: Plot correlations")
    plot_correlations(df)

    print("STEP 5: Hypothesis tests")
    run_hypothesis_tests(df)

    print("DONE ✅")


if __name__ == "__main__":
    main()