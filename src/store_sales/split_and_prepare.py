import pandas as pd
from sklearn.model_selection import train_test_split
from store_sales import *

data = pd.read_csv('../../data/train.csv')
holidays_events_data = pd.read_csv('../../data/holidays_events.csv')
oil_data = pd.read_csv('../../data/oil.csv')
stores_data = pd.read_csv('../../data/stores.csv')

X = data.drop(['sales'], axis=1)
y = data['sales']
X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, shuffle=False, test_size=0.5, random_state=42)

train_data = pd.concat([X_train, y_train], axis=1)
valid_data = pd.concat([X_valid, y_valid], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data = prepare_data(train_data, holidays_events_data, oil_data, stores_data)
valid_data = prepare_data(valid_data, holidays_events_data, oil_data, stores_data)
test_data = prepare_data(test_data, holidays_events_data, oil_data, stores_data)

train_data.to_csv("../../data/train_data.csv", index=False)
valid_data.to_csv("../../data/valid_data.csv", index=False)
test_data.to_csv("../../data/test_data.csv", index=False)
