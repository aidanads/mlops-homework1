# MLOps Homework 1

This project uses sampregdata to build linear regression models using scikit-learn. The previous model (v1.0) uses x4 as the single predictor, which had the highest R-squared among the four individual predictors, with R² = 0.275. The current model (v2.0) uses x3 and x4 as predictors and has R² = 0.530. The current two-predictor model therefore provides a better fit to the data than the previous one-predictor model.

## How to Run the Models

Install the required Python packages:

pip install -r requirements.txt

Run the previous one-predictor model:

python src/model_v1.py

Run the current two-predictor model:

python src/model_v2.py