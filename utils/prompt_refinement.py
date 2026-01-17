import pandas as pd
import os

def refine_prompt(user_message, file_name=None):
    """
    Refine the user prompt by adding dataset path and column info if available.
    Args:
        user_message (str): The original user prompt.
        file_name (str): Name of the uploaded dataset file (optional).
    Returns:
        str: The refined prompt.
    """
    # If the user didn't write anything, assume they just want to analyze the dataset
    if not user_message.strip() and file_name:
        refined = "Analyze the dataset"
    else:
        refined = user_message
    # Add file path info if available
    if file_name and os.path.isfile(f"data/raw/{file_name}"):
        refined += f"\nThe data file name is : {file_name}\nThe data path is data/raw/{file_name}"
        print("Refining prompt with file:", file_name)
        try:
            df = pd.read_csv(f"data/raw/{file_name}")
            columns = df.columns.tolist()
            refined += f"\nAvailable columns: {columns}"
            # Only suggest target column if prompt is about training or prediction
            lower_refined = refined.lower()
            if any(word in lower_refined for word in ["train", "predict", "regression", "classification"]):
                if 'target' not in user_message.lower() and len(columns) > 1:
                    refined += f"\nPlease specify the target column for training or prediction."
        except Exception:
            pass
    # If the prompt is about training or prediction, force tool usage
    lower_refined = refined.lower()
    if any(word in lower_refined for word in ["train", "predict", "regression", "classification", "model"]):
        refined += ("\nYou MUST use the available tools: train_models and predict_with_model for all training and prediction tasks. "
                    "Do not write manual training or prediction code.")
    return refined
