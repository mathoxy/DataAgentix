from smolagents import tool
from datasets import load_dataset
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os



# @tool
# def load_data(file_name: str) -> str:
#     """
#     Loads data from specified file or Hugging Face Hub.
#     Args:
#         file_name (str): The name of the file to load from Hugging Face Hub.
#     Returns:
#         str: The path to the loaded file.

#     """
#     os.makedirs("data/raw", exist_ok=True)
#     # If file exists locally, use it
#     if os.path.isfile(f"data/raw/{file_name}"):
#         return f"data/raw/{file_name}"
#     else:
#         # Load dataset from Hugging Face Hub and save to disk as csv
#         dataset = load_dataset(file_name)
#         df = dataset["train"].to_pandas()
#         dataset_path = f"data/raw/{file_name.replace('/', '_')}.csv"
#         df.to_csv(dataset_path, index=False)
#         return dataset_path
    
@tool
def dataset_summary_tool(dataset_path: str) -> dict:
    """
    Provides a summary of the dataset including number of rows, columns, missing values, duplicates, and descriptive statistics.
    Args:
        dataset_path (str): The path to the dataset file.
    Returns:
        dict: A summary of the dataset.
    """
    df = pd.read_csv(dataset_path)
    summary = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "descriptive_statistics": df.describe().to_dict()
    }
    return summary

@tool
def clean_dataset_tool(dataset_path: str) -> str:
    """
    Cleans the dataset by handling missing values and removing duplicates.
    Args:
        dataset_path (str): The path to the dataset file.
        Returns:
            str: The path to the cleaned dataset file.
    """
    os.makedirs("data/cleaned", exist_ok=True)
    df = pd.read_csv(dataset_path)

    # Duplicated values removal
    df = df.drop_duplicates()

    # Missing values handling - simpleImputer
    # mean for numerical columns
    num_cols = df.select_dtypes(include="number").columns
    importer = SimpleImputer(strategy='mean')
    df[num_cols] = importer.fit_transform(df[num_cols])
    # most_frequent for categorical columns
    cat_cols = df.select_dtypes(include="object").columns
    importer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = importer.fit_transform(df[cat_cols])
    
    # Save cleaned dataset
    cleaned_path = dataset_path.replace('raw', 'cleaned')
    df.to_csv(cleaned_path, index=False)
    return cleaned_path

@tool
def visualize_data_tool(dataset_path: str, target: str=None) -> list:
    """
    Generates separate plots to explore the dataset.
    Each plot is saved in a separate file.

    Args:
        dataset_path (str): The path to the dataset file.
        target (str, optional): The target variable for correlation plot. Defaults to None.
    Returns:
        list: Paths to the saved plots.
    """
    os.makedirs("data/plots", exist_ok=True)
    df = pd.read_csv(dataset_path)
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include="object").columns

    saved_files = []
    extracted_name = os.path.basename(dataset_path).replace('.csv','')
    # Numerical plots
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(10,5))
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Numerical Distribution: {col}")
        file_path = f"data/plots/{extracted_name}_{col}_num.png"
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        saved_files.append(file_path)

    # Categorical plots
    for col in cat_cols:
        fig, ax = plt.subplots(figsize=(10, max(5, len(df[col].unique())*0.3)))  # fit the height to number of categories
        sns.countplot(y=df[col], ax=ax)
        ax.set_title(f"Categorical Distribution: {col}")
        file_path = f"data/plots/{extracted_name}_{col}_cat.png"
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        saved_files.append(file_path)

    # Correlation matrix (si target spécifié)
    if target and target in df.columns:
        numeric_df = df.select_dtypes(include="number")
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Matrix")
        file_path = f"data/plots/{extracted_name}_correlation.png"
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        saved_files.append(file_path)

    return saved_files
