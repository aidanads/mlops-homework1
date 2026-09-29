import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load the data
data = pd.read_csv("data/sampregdata.csv")

# Select x4 as the predictor and y as the outcome
X = data[["x4"]]
y = data["y"]

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Calculate R-squared
r_squared = model.score(X, y)

# Display model information
print("Model 1")
print("Predictor: x4")
print("R-squared:", r_squared)
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])

# Save the model
joblib.dump(model, "models/model_v1.pkl")

print("Model saved as models/model_v1.pkl")