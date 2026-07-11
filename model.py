"""
model.py

Final Project - Part 1
Build, train, evaluate, and save a simple Linear Regression model.

Project topic:
Predict annual salary based on years of professional experience.
"""

from __future__ import annotations

import csv
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = BASE_DIR / "linear_regression_model.joblib"
X_FILENAME = BASE_DIR / "X.joblib"
Y_FILENAME = BASE_DIR / "y.joblib"
CSV_FILENAME = BASE_DIR / "salary_data.csv"


def generate_synthetic_salary_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic dataset for salary prediction.

    Feature:
        X = years of professional experience

    Target:
        y = annual salary in USD

    Approximate synthetic relationship:
        salary = 32000 + 5200 * years_of_experience + random_noise

    Returns:
        tuple[np.ndarray, np.ndarray]: feature matrix X and target values y.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    # 0 to 20 years of experience with 0.25-year steps.
    years_of_experience = np.arange(0, 20.25, 0.25)
    X = years_of_experience.reshape(-1, 1)

    base_salary = 32_000
    salary_growth_per_year = 5_200
    noise = rng.normal(loc=0, scale=6_000, size=len(years_of_experience))

    salary = base_salary + salary_growth_per_year * years_of_experience + noise
    salary = np.clip(salary, 25_000, None)

    return X, salary


def train_regression_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """
    Train a linear regression model.

    Args:
        X_train: Training feature matrix.
        y_train: Training target values.

    Returns:
        LinearRegression: trained regression model.
    """
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    return regression_model


def save_regression_model(
    regression_model: LinearRegression,
    filename: Path = MODEL_FILENAME,
) -> None:
    """
    Serialize and save the trained regression model.

    Args:
        regression_model: Trained LinearRegression model.
        filename: File path where the model will be saved.
    """
    joblib.dump(regression_model, filename)
    print(f"Model saved to {filename.name}")


def evaluate_regression_model(
    regression_model: LinearRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """
    Evaluate the regression model using Mean Squared Error.

    Args:
        regression_model: Trained regression model.
        X_test: Test feature matrix.
        y_test: Test target values.

    Returns:
        float: Mean Squared Error of the model predictions.
    """
    predictions = regression_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")

    return mse


def save_initial_datasets(
    X: np.ndarray,
    y: np.ndarray,
    X_filename: Path = X_FILENAME,
    y_filename: Path = Y_FILENAME,
) -> None:
    """
    Serialize and save the original feature matrix and target values.

    Args:
        X: Complete feature matrix.
        y: Complete target values.
        X_filename: File path where X will be saved.
        y_filename: File path where y will be saved.
    """
    joblib.dump(X, X_filename)
    joblib.dump(y, y_filename)

    print(f"Feature matrix saved to {X_filename.name}")
    print(f"Target values saved to {y_filename.name}")


def save_dataset_as_csv(X: np.ndarray, y: np.ndarray, filename: Path = CSV_FILENAME) -> None:
    """
    Save a readable CSV copy of the generated synthetic dataset.

    This file is not required by the assignment, but it makes the generated data
    easier to inspect.

    Args:
        X: Complete feature matrix.
        y: Complete target values.
        filename: File path where the CSV dataset will be saved.
    """
    with filename.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["YearsExperience", "Salary"])
        for years, salary in zip(X.ravel(), y):
            writer.writerow([f"{years:.2f}", f"{salary:.2f}"])

    print(f"Readable CSV dataset saved to {filename.name}")


def main() -> None:
    """Run the full model development pipeline."""
    X, y = generate_synthetic_salary_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    regression_model = train_regression_model(X_train, y_train)
    evaluate_regression_model(regression_model, X_test, y_test)

    save_regression_model(regression_model)
    save_initial_datasets(X, y)
    save_dataset_as_csv(X, y)

    print("\nProject topic: Salary prediction by years of experience")
    print(
        "Learned model formula: "
        f"salary = {regression_model.intercept_:.2f} "
        f"+ {regression_model.coef_[0]:.2f} * years_of_experience"
    )


if __name__ == "__main__":
    main()
