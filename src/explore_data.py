import pandas as pd

data = pd.read_csv("data/sampregdata.csv")

print(data.head())
print(data.columns)
print(data.shape)