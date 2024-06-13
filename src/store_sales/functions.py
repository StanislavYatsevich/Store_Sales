import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def prepare_data(data, holidays_events_data, oil_data, stores_data):
    holidays_events_data['priority'] = holidays_events_data['locale'].map({'National': 3, 'Regional': 2, 'Local': 1})
    holidays_events_data = holidays_events_data.sort_values(by=['date', 'priority'], ascending=False)
    holidays_events_data = holidays_events_data.drop_duplicates(subset=['date'], keep='first')
    holidays_events_data.drop('priority', axis=1, inplace=True)

    data = pd.merge(data, stores_data, on=['store_nbr'], how='inner')
    data = pd.merge(data, oil_data, on=['date'], how='left')
    data = pd.merge(data, holidays_events_data, on=['date'], how='left')
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
    
    data = data.sort_values(by=['store_number', 'item_family', 'date'])
    data['mean_sales_prev_week'] = data.groupby(['store_number', 'item_family'])['item_sales'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
    data['mean_sales_prev_week'] = data['mean_sales_prev_week'].fillna(method='bfill')
    data = data.sort_values(by=['date', 'store_number', 'item_family'])
    return data


def add_features(data):
    def is_during_falling_period(date, periods):
        for start, end in periods:
            if start <= date <= end:
                return 1
        return 0
    
    data['date'] = pd.to_datetime(data['date'])
    periods = [(pd.to_datetime('2014-07-01'), pd.to_datetime('2015-01-31')), (pd.to_datetime('2015-06-01'), pd.to_datetime('2016-02-29'))]
    data['is_during_oil_prices_falling'] = data['date'].apply(lambda x: is_during_falling_period(x, periods))


    def is_popular_unit(unit, popular_list):
        if unit in popular_list:
            return 1
        return 0

    data['is_popular_store'] = data['store_number'].apply(lambda x: is_popular_unit(x, [3, 8, 11, 44, 45, 46, 47, 48, 49, 50, 51]))
    data['is_popular_cluster'] = data['store_cluster'].apply(lambda x: is_popular_unit(x, [5, 8, 11, 14, 17]))
    data['is_special_non_working_day'] = data['day_type'].apply(lambda x: is_popular_unit(x, ['Additional', 'Bridge', 'Transfer', 'Event']))
    data['is_national_holiday'] = data['holiday_status'].apply(lambda x: is_popular_unit(x, ['National']))
    data['is_state_pichincha'] = data['state'].apply(lambda x: is_popular_unit(x, ['Pichincha']))
    data['is_city_quito_or_cayambe'] = data['city'].apply(lambda x: is_popular_unit(x, ['Quito', 'Cayambe']))
    data['is_store_type_A'] = data['store_type'].apply(lambda x: is_popular_unit(x, ['A']))
    data['number_of_days_since_earthquake'] = (data['date'] - pd.to_datetime('2016-04-16')).dt.days
    return data


def encode_features(train_data, test_data):
    min_date = pd.to_datetime(train_data['date']).min()

    def add_date_features(data):
        data['date'] = pd.to_datetime(data['date'])
        data['days_since_start'] = (pd.to_datetime(data['date']) - min_date).dt.days
        data['year'] = pd.to_datetime(data['date']).dt.year
        data['month'] = pd.to_datetime(data['date']).dt.month
        data['day_of_week'] = pd.to_datetime(data['date']).dt.dayofweek
        data.drop(['date'], axis=1, inplace=True)
        return data
    
    train_data = add_date_features(train_data)
    test_data = add_date_features(test_data)

    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    cat_columns = ['item_family', 'city', 'state', 'store_type', 'day_type', 'holiday_status',
                'holiday_location', 'holiday_description', 'is_holiday_transferred']
    
    train_data[cat_columns] = ordinal_encoder.fit_transform(train_data[cat_columns])
    test_data[cat_columns] = ordinal_encoder.transform(test_data[cat_columns])

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


def get_mae_score_cross_validation(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = pd.Series(model.predict(X_test), index=y_test.index)
    mae_score = np.round(mean_absolute_error(y_test, y_pred), 2)
    return mae_score

