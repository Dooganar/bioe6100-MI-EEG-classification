import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

channels = "8 Channel Data"

# Load the CSV file
file_path = "./models-8ch-tasks12-200epoch-test-accuracys.csv"
df = pd.read_csv(file_path)

df['Difference'] = df['Task 1'] - df['Task 2']

# Show the first few rows and column names to understand the structure
df.head(), df.columns

# Set a consistent style
sns.set(style="whitegrid")

from scipy.stats import ttest_rel
import numpy as np

# Compute mean and 95% confidence interval for the difference
mean_diff = df['Difference'].mean()
std_diff = df['Difference'].std(ddof=1)
n = len(df)
sem_diff = std_diff / np.sqrt(n)
ci_low = mean_diff - 1.96 * sem_diff
ci_high = mean_diff + 1.96 * sem_diff

# Compute Cohen's d
cohens_d = mean_diff / std_diff


# Check for NaNs or constant rows
nan_rows = df[['Task 1', 'Task 2']].isna().any(axis=1)
constant_rows = (df['Task 1'] == df['Task 2'])

# Count and filter them out
num_nan = nan_rows.sum()
num_constant = constant_rows.sum()

# Filter clean data for valid t-test
clean_df = df[~nan_rows]

# Run paired t-test on clean data
t_stat_clean, p_value_clean = ttest_rel(clean_df['Task 1'], clean_df['Task 2'])

num_nan, num_constant, t_stat_clean, p_value_clean



def plot_analysis():
    # Create histogram of the differences
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Difference'], kde=True, bins=20)
    plt.title("Distribution of Accuracy Differences (MM - MI) - " + channels)
    plt.xlabel("Accuracy Difference (%)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Create boxplots for Task 1 and Task 2
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df[['Task 1', 'Task 2']])
    plt.title("Accuracy Comparison: MM vs MI - " + channels)
    plt.ylabel("Accuracy (%)")
    plt.xticks([0, 1], ['Motor Movement (Task 1)', 'Motor Imagery (Task 2)'])
    plt.tight_layout()
    plt.show()

    # Scatter plot of Task 1 vs Task 2 with identity line
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x='Task 2', y='Task 1', data=df)
    plt.plot([0, 100], [0, 100], color='red', linestyle='--')  # Identity line
    plt.xlabel("Motor Imagery Accuracy (Task 2)")
    plt.ylabel("Motor Movement Accuracy (Task 1)")
    plt.title("Scatter Plot: Motor Movement vs Motor Imagery - " + channels)
    plt.tight_layout()
    plt.show()

def print_analysis():
    print("---------------- Statistics for", channels, "---------------------")

    print("num_nan: ", num_nan)
    print("num_constant: ", num_constant)

    print("t_stat_clean: ", t_stat_clean, "| p_value_clean: ", p_value_clean)
        
    print("mean_diff: ", mean_diff)
    print("95% Confidence Interval: ", (float(ci_low), float(ci_high)))
    print("cohens_d: ", cohens_d)

    print("------------------------------------------------------------------")


def output_data_to_latex_format():
    # Load the data files
    path_64ch = "./models-64ch-tasks12-200epoch-test-accuracys.csv"
    path_8ch = "./models-8ch-tasks12-200epoch-test-accuracys.csv"

    df_64ch = pd.read_csv(path_64ch)
    df_8ch = pd.read_csv(path_8ch)

    # Merge the two datasets on 'Subject' to prepare a LaTeX table
    merged_df = df_64ch.merge(df_8ch, on="Subject", suffixes=("_64ch", "_8ch"))

    # Rename columns for clarity
    merged_df.columns = [
        "Subject", 
        "Task 1 (64ch)", 
        "Task 2 (64ch)", 
        "Task 1 (8ch)", 
        "Task 2 (8ch)"
    ]

    for _, row in merged_df.iterrows():
        formatted_row = f"{int(row['Subject'])} & {row['Task 1 (64ch)']:.2f} & {row['Task 2 (64ch)']:.2f} & {row['Task 1 (8ch)']:.2f} & {row['Task 2 (8ch)']:.2f} \\\\"
        print(formatted_row)

    # Compute and print averages
    avg_row = merged_df.iloc[:, 1:].mean()
    formatted_avg_row = f"\\textbf{{Average}} & {avg_row['Task 1 (64ch)']:.2f} & {avg_row['Task 2 (64ch)']:.2f} & {avg_row['Task 1 (8ch)']:.2f} & {avg_row['Task 2 (8ch)']:.2f} \\\\"
    print("\\midrule")
    print(formatted_avg_row)
    

if __name__ == "__main__":
    plot_analysis()
    print_analysis()
    # output_data_to_latex_format()
