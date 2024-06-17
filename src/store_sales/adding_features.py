import pandas as pd
from store_sales import add_features

train_data = pd.read_csv("../../data/prepared_data/train_data.csv")
valid_data = pd.read_csv("../../data/prepared_data/valid_data.csv")
test_data = pd.read_csv("../../data/prepared_data/test_data.csv")

data = pd.concat([train_data, valid_data, test_data], axis=0)
data = add_features(data)
data.to_csv("../../data/prepared_data/prepared_data.csv", index=False)