"""
app.py

Final Project - Part 2
Streamlit application for serving a trained Linear Regression model.

Project topic:
Predict annual salary based on years of professional experience.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILENAME = BASE_DIR / "linear_regression_model.joblib"
X_FILENAME = BASE_DIR / "X.joblib"
Y_FILENAME = BASE_DIR / "y.joblib"


@st.cache_resource
def load_model(model_filename: Path = MODEL_FILENAME):
    """
    Load the trained regression model from disk.

    Args:
        model_filename: Path to the saved model file.

    Returns:
        Loaded regression model.
    """
    return joblib.load(model_filename)


@st.cache_data
def load_initial_datasets(
    X_filename: Path = X_FILENAME,
    y_filename: Path = Y_FILENAME,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the original synthetic dataset from disk.

    Args:
        X_filename: Path to the saved feature matrix.
        y_filename: Path to the saved target values.

    Returns:
        tuple[np.ndarray, np.ndarray]: feature matrix X and target values y.
    """
    X = joblib.load(X_filename)
    y = joblib.load(y_filename)
    return X, y


def load_and_predict(
    input_feature: float,
    model_filename: Path = MODEL_FILENAME,
) -> float:
    """
    Load a saved regression model and make a prediction.

    Args:
        input_feature: User-selected years of experience.
        model_filename: Path to the saved model file.

    Returns:
        float: Predicted annual salary.
    """
    regression_model = load_model(model_filename)
    input_array = np.array([[input_feature]], dtype=float)
    prediction = regression_model.predict(input_array)
    return float(prediction[0])


def find_actual_target(
    input_feature: float,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    """
    Find the actual target value in the dataset closest to the selected feature.

    The Streamlit slider uses the same 0.25-year step as the generated dataset,
    so the selected value usually exists exactly in X. The nearest-value fallback
    makes the app robust to floating-point differences.

    Args:
        input_feature: User-selected years of experience.
        X: Original feature matrix.
        y: Original target values.

    Returns:
        tuple[float, float]: actual feature value and actual salary.
    """
    x_values = X.ravel()
    nearest_index = int(np.abs(x_values - input_feature).argmin())
    return float(x_values[nearest_index]), float(y[nearest_index])


def visualize_difference(input_feature: float, prediction: float) -> tuple[plt.Figure, float, float, float]:
    """
    Visualize the dataset, actual target, predicted target, and difference.

    Args:
        input_feature: User-selected years of experience.
        prediction: Predicted annual salary.

    Returns:
        tuple containing figure, actual feature value, actual salary, and difference.
    """
    X, y = load_initial_datasets()
    actual_x, actual_y = find_actual_target(input_feature, X, y)
    difference = prediction - actual_y

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(X.ravel(), y, color="gray", alpha=0.65, label="Dataset")
    ax.scatter(actual_x, actual_y, color="blue", s=120, label="Actual Target")
    ax.scatter(input_feature, prediction, color="red", s=120, label="Predicted Target")

    ax.plot(
        [input_feature, input_feature],
        [actual_y, prediction],
        color="black",
        linestyle="--",
        linewidth=2,
    )

    middle_y = (actual_y + prediction) / 2
    ax.annotate(
        f"Difference = ${difference:,.2f}",
        xy=(input_feature, middle_y),
        xytext=(input_feature + 0.8, middle_y),
        arrowprops={"arrowstyle": "->"},
        fontsize=10,
    )

    ax.set_title("Prediction vs Actual Target")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Annual Salary, USD")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    return fig, actual_x, actual_y, difference


def create_streamlit_app() -> None:
    """Create the Streamlit web application user interface."""
    st.set_page_config(page_title="Salary Prediction", page_icon="💼", layout="centered")

    st.title("Salary Prediction by Years of Experience")
    st.write(
        "This app uses a simple Linear Regression model to predict annual salary "
        "based on years of professional experience."
    )

    required_files = [MODEL_FILENAME, X_FILENAME, Y_FILENAME]
    missing_files = [file.name for file in required_files if not file.exists()]

    if missing_files:
        st.error(
            "Missing required files: "
            + ", ".join(missing_files)
            + ". Please run `python model.py` first."
        )
        return

    years_of_experience = st.slider(
        "Years of Experience for Prediction",
        min_value=0.0,
        max_value=20.0,
        value=5.0,
        step=0.25,
    )

    if st.button("Predict value"):
        prediction = load_and_predict(years_of_experience)
        st.success(
            f"Prediction for {years_of_experience:.2f} years of experience: "
            f"${prediction:,.2f} per year"
        )

        fig, actual_x, actual_y, difference = visualize_difference(
            years_of_experience,
            prediction,
        )
        st.pyplot(fig)
        plt.close(fig)

        st.info(
            f"Actual dataset point: {actual_x:.2f} years of experience, "
            f"actual salary ${actual_y:,.2f}. "
            f"Prediction difference: ${difference:,.2f}."
        )


if __name__ == "__main__":
    create_streamlit_app()
