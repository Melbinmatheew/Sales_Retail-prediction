import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os

# Create dummy data
X = np.random.rand(100, 1)
y = 2 * X.squeeze() + np.random.rand(100)

# Train a simple model
model = LinearRegression()
model.fit(X, y)

# Save the model
model_dir = '/app/models'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'best_sales_forecasting_model_sarima.joblib')
joblib.dump(model, model_path)

print(f"Dummy model saved to {model_path}")
