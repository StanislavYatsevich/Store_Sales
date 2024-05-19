import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def prepare_data(data, holidays_events_data, oil_data, stores_data):
    holidays_events_data['priority'] = holidays_events_data['locale'].map({'National': 3, 'Regional': 2, 'Local': 1})
    holidays_events_data = holidays_events_data.sort_values(by=['date', 'priority'], ascending=False)
    holidays_events_data = holidays_events_data.drop_duplicates(subset=['date'], keep='first')
    holidays_events_data.drop('priority', axis=1, inplace=True)

    data = pd.merge(data, stores_data, on=['store_nbr'], how='inner')
    data = pd.merge(data, oil_data, on=['date'], how='left')
    data = pd.merge(data, holidays_events_data, on=['date'], how='left')
    #data = pd.merge(data, transactions_data, on=['date', 'store_nbr'], how='left')   
    data.fillna({'type_y': 'Not holiday', 'locale': 'Not holiday', 'locale_name' : 'Not holiday', 'description' : 'Not holiday',
                         'transferred' : 'Not holiday'}, inplace=True)

    data.rename(columns={'store_nbr' : 'store_number', 'type_x': 'store_type', 'cluster' : 'store_cluster',
                        'dcoilwtico' : 'oil_price', 'locale' : 'holiday_status', 'locale_name' : 'holiday_location',
                        'description' : 'holiday_description', 'type_y' : 'day_type', 'transferred' : 'is_holiday_transferred',
                        'family' : 'item_family', 'sales' : 'item_sales', 'onpromotion' : 'items_on_promotion'}, inplace=True)
    data['date'] = pd.to_datetime(data['date'])  
    data.set_index('id', inplace=True)
    data['oil_price'].bfill(inplace=True)
    data['is_holiday_transferred'] = data['is_holiday_transferred'].map(lambda x: False if x == False or x == 'Not holiday' else True)
    return data



def encode_features(train_data, test_data):
    train_data['days_since_start'] = (train_data['date'] - pd.to_datetime('2013-01-01')).dt.days
    train_data['date'] = train_data['days_since_start']
    train_data.drop(['days_since_start'], axis=1, inplace=True)
    train_data.rename(columns={'date': 'days_since_start'}, inplace=True)

    test_data['days_since_start'] = (test_data['date'] - pd.to_datetime('2013-01-01')).dt.days
    test_data['date'] = test_data['days_since_start']
    test_data.drop(['days_since_start'], axis=1, inplace=True)
    test_data.rename(columns={'date': 'days_since_start'}, inplace=True)

    label_encoder = LabelEncoder()
    train_data['item_family'] = label_encoder.fit_transform(train_data['item_family'])
    test_data['item_family'] = label_encoder.transform(test_data['item_family'])

    train_data['city'] = label_encoder.fit_transform(train_data['city'])
    test_data['city'] = label_encoder.transform(test_data['city'])

    train_data['state'] = label_encoder.fit_transform(train_data['state'])
    test_data['state'] = label_encoder.transform(test_data['state'])

    train_data['store_type'] = label_encoder.fit_transform(train_data['store_type'])
    test_data['store_type'] = label_encoder.transform(test_data['store_type'])

    train_data['day_type'] = label_encoder.fit_transform(train_data['day_type'])
    test_data['day_type'] = label_encoder.transform(test_data['day_type'])

    train_data['holiday_status'] = label_encoder.fit_transform(train_data['holiday_status'])
    test_data['holiday_status'] = label_encoder.transform(test_data['holiday_status'])

    train_data['holiday_location'] = label_encoder.fit_transform(train_data['holiday_location'])
    test_data['holiday_location'] = label_encoder.transform(test_data['holiday_location'])

    train_data['holiday_description'] = label_encoder.fit_transform(train_data['holiday_description'])
    test_data['holiday_description'] = label_encoder.transform(test_data['holiday_description'])

    train_data['is_holiday_transferred'] = label_encoder.fit_transform(train_data['is_holiday_transferred'])
    test_data['is_holiday_transferred'] = label_encoder.transform(test_data['is_holiday_transferred'])

    return train_data, test_data



def get_tree_based_predicts(X_train, X_test, y_train, y_test, models_list, models_names):
    df = pd.concat([y_train, y_test])
    mae_scores = []
    mape_scores = []

    for model in models_list:
        model.fit(X_train, y_train)
        y_pred = pd.Series(model.predict(X_test), index=y_test.index)
        predict = pd.concat([y_train, y_pred])
        df = pd.concat([df, predict], axis=1)
        mae_scores.append(np.round(mean_absolute_error(y_test, y_pred), 2))
        mape_scores.append(100 * np.round(mean_absolute_percentage_error(y_test, y_pred), 2))

    df.columns = ['Real Sales'] + models_names
    df = pd.concat([pd.concat([X_train, X_test], axis=0), df], axis=1)

    return (df, mae_scores, mape_scores)

