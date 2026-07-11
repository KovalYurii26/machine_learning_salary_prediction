# Salary Prediction by Years of Experience

Completed final project variant: a simple machine learning project that trains a Linear Regression model and serves it through a Streamlit web application.

## Project topic

The project predicts annual salary based on years of professional experience.

- Input feature: years of experience
- Target value: annual salary in USD
- Model: Linear Regression
- Dataset type: synthetic dataset generated in Python

## Files

```text
model.py
app.py
requirements.txt
linear_regression_model.joblib
X.joblib
y.joblib
salary_data.csv
```

## How the synthetic dataset is generated

The dataset is generated using this approximate formula:

```text
salary = 32000 + 5200 * years_of_experience + random_noise
```

The feature values go from 0 to 20 years of experience with 0.25-year steps.

## How to run the project

Install dependencies:

```bash
pip install -r requirements.txt
```

Train and save the model:

```bash
python model.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## What the app does

The Streamlit app allows the user to choose years of experience with a slider and click **Predict value**. The app then:

1. loads the saved Linear Regression model;
2. predicts annual salary;
3. loads the original synthetic dataset;
4. finds the actual salary value for the selected experience value;
5. shows a chart with the full dataset, actual salary, predicted salary, and the difference between them.
