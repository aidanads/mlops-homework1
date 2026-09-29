import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data/sampregdata.csv")

y = data["y"]

for column in ["x1", "x2", "x3", "x4"]:
    X = data[[column]]

    model = LinearRegression()
    model.fit(X, y)

    score = model.score(X, y)

    print(column, score)