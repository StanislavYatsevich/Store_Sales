import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from typing import List, Any, Tuple
from store_sales import NOT_HOLIDAY_DAY


def prepare_data(
    data: pd.DataFrame,
    holidays_events_data: pd.DataFrame,
    oil_data: pd.DataFrame,
    stores_data: pd.DataFrame,
) -> pd.DataFrame:
    """Collects and prepares raw data.

    Collects raw data by merging it together. Prepares features and renames
    them. Sorts data in the ascending order.

    Args:
        data: the pd.DataFrame instance with main sales data.
        holidays_events_data: the pd.DataFrame instance with holidays data.
        oil_data: the pd.DataFrame instance with oil prices data.
        stores_data: the pd.DataFrame instance with stores data.

    Returns:
        pd.DataFrame instance with prepared data.
    """
    holidays_events_data["priority"] = holidays_events_data["locale"].map(
        {"National": 3, "Regional": 2, "Local": 1}
    )
    holidays_events_data = holidays_events_data.sort_values(
        by=["date", "priority"], ascending=False
    )
    holidays_events_data = holidays_events_data.drop_duplicates(
        subset=["date"], keep="first"
    )
    holidays_events_data.drop("priority", axis=1, inplace=True)

    data = pd.merge(data, stores_data, on=["store_nbr"], how="inner")
    data = pd.merge(data, oil_data, on=["date"], how="left")
    data = pd.merge(data, holidays_events_data, on=["date"], how="left")

    columns_to_fill = ["type_y", "locale", "locale_name", "description", "transferred"]
    data.fillna({column: NOT_HOLIDAY_DAY for column in columns_to_fill}, inplace=True)

    data.rename(
        columns={
            "store_nbr": "store_number",
            "type_x": "store_type",
            "cluster": "store_cluster",
            "dcoilwtico": "oil_price",
            "locale": "holiday_status",
            "locale_name": "holiday_location",
            "description": "holiday_description",
            "type_y": "day_type",
            "transferred": "is_holiday_transferred",
            "family": "item_family",
            "sales": "item_sales",
            "onpromotion": "items_on_promotion",
        },
        inplace=True,
    )
    data["date"] = pd.to_datetime(data["date"])
    data.set_index("date", inplace=True)
    data["oil_price"].interpolate(method="time", inplace=True)

    data.reset_index(inplace=True)
    data.set_index("id", inplace=True)
    data["is_holiday_transferred"] = data["is_holiday_transferred"].map(
        lambda x: False if not x or x == "Not holiday" else True
    )

    data.sort_values(by=["store_number", "item_family", "date"], inplace=True)
    data["mean_sales_prev_month"] = data.groupby(["store_number", "item_family"])[
        "item_sales"
    ].transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean())
    data["mean_sales_prev_month"].fillna(method="bfill", inplace=True)
    data.sort_values(by=["date", "store_number", "item_family"], inplace=True)
    return data


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Adds new features to data.

    Adds new binary features depending on the values of some other
    features. Adds the "number_of_days_since_earthquake" feature.

    Args:
        data: the pd.DataFrame instance with sales data.

    Returns:
        pd.DataFrame instance with new features added.
    """

    def is_during_falling_periods(date, periods):
        """Checks whether a given date falls within a given period."""
        for start, end in periods:
            if start <= date <= end:
                return 1
        return 0

    oil_price_falling_start_1 = pd.to_datetime("2014-07-01")
    oil_price_falling_finish_1 = pd.to_datetime("2015-01-31")
    oil_price_falling_start_2 = pd.to_datetime("2015-06-01")
    oil_price_falling_finish_2 = pd.to_datetime("2016-02-29")
    data["date"] = pd.to_datetime(data["date"])
    periods = [
        (oil_price_falling_start_1, oil_price_falling_finish_1),
        (oil_price_falling_start_2, oil_price_falling_finish_2),
    ]
    data["is_during_oil_prices_falling"] = data["date"].apply(
        lambda x: is_during_falling_periods(x, periods)
    )

    def is_special_unit(unit: Any, special_list: List[Any]) -> int:
        if unit in special_list:
            return 1
        return 0

    popular_stores = [3, 8, 11, 44, 45, 46, 47, 48, 49, 50, 51]
    popular_clusters = [5, 8, 11, 14, 17]
    non_popular_clusters = [7]
    special_non_working_days = ["Additional", "Bridge", "Transfer", "Event"]
    popular_holidays = ["National"]
    popular_states = ["Pichincha"]
    non_popular_states = ["Manabi", "Pastaza"]
    popular_cities = ["Quito", "Cayambe"]
    non_popular_cities = ["Manta", "Puyo"]
    popular_store_types = ["A"]
    date_of_earthquake = pd.to_datetime("2016-04-16")

    data["is_popular_store"] = data["store_number"].apply(
        lambda x: is_special_unit(x, popular_stores)
    )
    data["is_popular_cluster"] = data["store_cluster"].apply(
        lambda x: is_special_unit(x, popular_clusters)
    )
    data["is_non_popular_cluster"] = data["store_cluster"].apply(
        lambda x: is_special_unit(x, non_popular_clusters)
    )
    data["is_special_non_working_day"] = data["day_type"].apply(
        lambda x: is_special_unit(x, special_non_working_days)
    )
    data["is_national_holiday"] = data["holiday_status"].apply(
        lambda x: is_special_unit(x, popular_holidays)
    )
    data["is_state_pichincha"] = data["state"].apply(
        lambda x: is_special_unit(x, popular_states)
    )
    data["is_state_manabi_or_pastaza"] = data["state"].apply(
        lambda x: is_special_unit(x, non_popular_states)
    )
    data["is_city_quito_or_cayambe"] = data["city"].apply(
        lambda x: is_special_unit(x, popular_cities)
    )
    data["is_city_manta_or_puyo"] = data["city"].apply(
        lambda x: is_special_unit(x, non_popular_cities)
    )
    data["is_store_type_A"] = data["store_type"].apply(
        lambda x: is_special_unit(x, popular_store_types)
    )
    data["number_of_days_since_earthquake"] = (
        data["date"] - date_of_earthquake
    ).dt.days
    return data


def encode_features(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Encodes features in the data.

    Adds some new data features and encodes existing ones with
    OrdinalEncoder().

    Args:
        train_data: pd.DataFrame instance representing the train part of the data.
        test_data: pd.DataFrame instance representing the test part of the data.

    Returns:
        Tuple(train_data, test_data) where train_data and test_data are pd.DataFrame
        instances with added and encoded features.
    """
    min_date = pd.to_datetime(train_data["date"]).min()

    def add_date_features(data: pd.DataFrame) -> pd.DataFrame:
        """Adds some new date features to the data."""
        data["date"] = pd.to_datetime(data["date"])
        data["days_since_start"] = (pd.to_datetime(data["date"]) - min_date).dt.days
        data["year"] = pd.to_datetime(data["date"]).dt.year
        data["month"] = pd.to_datetime(data["date"]).dt.month
        data["day_of_week"] = pd.to_datetime(data["date"]).dt.dayofweek
        data.drop(["date"], axis=1, inplace=True)
        return data

    train_data = add_date_features(train_data)
    test_data = add_date_features(test_data)

    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )

    cat_columns = [
        "item_family",
        "city",
        "state",
        "store_type",
        "day_type",
        "holiday_status",
        "holiday_location",
        "holiday_description",
        "is_holiday_transferred",
    ]

    train_data[cat_columns] = ordinal_encoder.fit_transform(train_data[cat_columns])
    test_data[cat_columns] = ordinal_encoder.transform(test_data[cat_columns])

    return train_data, test_data