import pandas as pd
from sklearn.linear_model import LinearRegression
from itertools import combinations

data = pd.read_csv("data/sampregdata.csv")

y = data["y"]

predictors = ["x1", "x2", "x3", "x4"]

for pair in combinations(predictors, 2):
    X = data[list(pair)]

    model = LinearRegression()
    model.fit(X, y)

    score = model.score(X, y)

    print(pair, score)