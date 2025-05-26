# Retail Sales Forecasting

## Objective
Predict future sales for a retail superstore to optimize inventory and inform business strategy, using time series analysis and machine learning.

## Tech Stack
- Python
- Key Libraries:
    - pandas
    - numpy
    - scikit-learn (sklearn)
    - statsmodels
    - plotly
    - dash
    - matplotlib
    - seaborn
    - joblib
    - pmdarima

## Dataset
Superstore Sales Dataset.
- Source: [https://www.kaggle.com/datasets/bhanupratapbiswas/superstore-salest](https://www.kaggle.com/datasets/bhanupratapbiswas/superstore-salest)
- The actual file used in this project is `data/superstore_final_dataset.csv`.

## Project Structure
```
├── data/
│   ├── superstore_final_dataset.csv
│   ├── cleaned_sales_data.csv
│   └── feature_engineered_sales_data.csv
├── models/
│   └── best_sales_forecasting_model_sarima.joblib
├── notebooks/
│   ├── 01_data_exploration_preprocessing.ipynb
│   ├── 02_eda_feature_engineering.ipynb
│   ├── 03_model_training_evaluation_part1.ipynb
│   ├── 04_model_training_evaluation_part2.ipynb
│   └── 05_future_forecasting.ipynb
├── dashboard.py
├── create_dummy_model.py
├── requirements.txt
└── README.md
```

## Setup Instructions
1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

### Jupyter Notebooks
Open and run the notebooks in the `notebooks/` directory sequentially (01 to 05) to understand the data processing, analysis, model training, and forecasting steps.
You can use Jupyter Lab or Jupyter Notebook:
```bash
jupyter lab
# or
jupyter notebook
```

### Dashboard
Run the dashboard application from the terminal:
```bash
python dashboard.py
```
The dashboard will typically be accessible at `http://127.0.0.1:8050/` (or the address shown in your terminal).

### Forecasting
Use `notebooks/05_future_forecasting.ipynb` to generate sales forecasts for a specified future period using the best trained model. Modify the `forecast_periods` variable in the notebook to set the desired forecast length.

## Summary of Findings
The project involved exploring historical sales data, engineering relevant time-based and lag features, and evaluating several forecasting models. These included baseline models (Naive, Seasonal Naive), regression models (Linear Regression, Decision Tree, Random Forest), and advanced time series models (SARIMA, Holt-Winters Exponential Smoothing).

The best performing model was identified as **SARIMA** (based on the filename `best_sales_forecasting_model_sarima.joblib`, though a dummy model is currently in place). This model (or its placeholder) achieved an RMSE of **[Placeholder Value - e.g., from Notebook 04 output]** and MAE of **[Placeholder Value - e.g., from Notebook 04 output]** on the walk-forward validation test set.

This model is used for generating future sales predictions in `notebooks/05_future_forecasting.ipynb` and is intended to be visualized in the `dashboard.py` application. The dashboard provides an interactive way to explore sales trends and (illustrative) model forecasts.

*(Note: Specific performance metrics for the best model should be retrieved from the output of `notebooks/04_model_training_evaluation_part2.ipynb` after a full execution run.)*
