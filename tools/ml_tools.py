from smolagents import tool
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

@tool
def train_models(cleaned_dataset_path: str, target: str, task_type: str) -> dict:
    """
    A simple model training pipeline that trains a model and evaluates it.
    Args:
        cleaned_dataset_path (str): The path to the cleaned dataset file.
        target (str): The target variable for prediction.
        task_type (str): The type of ML task ('classification' or 'regression').
    Returns:
        dict: A dictionary containing model performance metrics and trained models paths.
    """
    # Direcrtory to save models
    model_dir = "models/"
    os.makedirs(model_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(cleaned_dataset_path)

    # Categorical encoding (simple approach)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes


    # Split dataset into features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model based on task type
    # if regression train LinearRegression or RandomForestRegressor
    if task_type == 'regression':
        model1 = LinearRegression()
        model2 = RandomForestRegressor(n_estimators=100, random_state=42)
        # Train models
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)
        # Evaluate models
        preds1 = model1.predict(X_test)
        preds2 = model2.predict(X_test)
        mse1 = ((y_test - preds1) ** 2).mean()
        mse2 = ((y_test - preds2) ** 2).mean()
        # Save models
        joblib.dump(model1, os.path.join(model_dir, "linear_regression_model.joblib"))
        joblib.dump(model2, os.path.join(model_dir, "random_forest_regressor_model.joblib"))
        return {
            "LinearRegression": {"MSE": mse1, "model_path": os.path.join(model_dir, "linear_regression_model.joblib")},
            "RandomForestRegressor": {"MSE": mse2, "model_path": os.path.join(model_dir, "random_forest_regressor_model.joblib")},
        }
    else:  # classification
        model1 = RandomForestClassifier(n_estimators=100, random_state=42)
        model2 = LogisticRegression(max_iter=200)
        # Train models
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)
        # Evaluate models
        preds1 = model1.predict(X_test)
        preds2 = model2.predict(X_test)
        acc1 = accuracy_score(y_test, preds1)
        acc2 = accuracy_score(y_test, preds2)
        report1 = classification_report(y_test, preds1, output_dict=True)
        report2 = classification_report(y_test, preds2, output_dict=True)
        # Save models
        joblib.dump(model1, os.path.join(model_dir, "random_forest_classifier_model.joblib"))
        joblib.dump(model2, os.path.join(model_dir, "logistic_regression_model.joblib"))
        return {
            "RandomForestClassifier": {"Accuracy": acc1, "model_path": os.path.join(model_dir, "random_forest_classifier_model.joblib")},
            "LogisticRegression": {"Accuracy": acc2, "model_path": os.path.join(model_dir, "logistic_regression_model.joblib")},
        }
    
@tool
def predict_with_model(model_path: str, input_data: dict) -> any:
    """
    Load a trained model and make predictions on new input data.
    Args:
        model_path (str): The path to the trained model file.
        input_data (dict): A dictionary of input features for prediction.
    Returns:
        any: The prediction result from the model.
    """
    # Load the model
    model = joblib.load(model_path)
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    # Make prediction
    prediction = model.predict(input_df)
    return prediction.tolist()
    
