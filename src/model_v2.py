import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load the data
data = pd.read_csv("data/sampregdata.csv")

# Select x3 and x4 as predictors and y as the outcome
X = data[["x3", "x4"]]
y = data["y"]

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Calculate R-squared
r_squared = model.score(X, y)

# Display model information
print("Model 2")
print("Predictors: x3 and x4")
print("R-squared:", r_squared)
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Save the model
joblib.dump(model, "models/model_v2.pkl")

print("Model saved as models/model_v2.pkl")