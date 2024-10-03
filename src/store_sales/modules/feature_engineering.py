import pandas as pd
from typing import List, Any, Tuple
from store_sales.modules import (
    NOT_HOLIDAY_DAY,
    SPECIAL_NON_WORKING_DAYS,
    POPULAR_HOLIDAYS,
    DATE_OF_EARTHQUAKE,
    OIL_PRICE_FALLING_START_1,
    OIL_PRICE_FALLING_FINISH_1,
    OIL_PRICE_FALLING_START_2,
    OIL_PRICE_FALLING_FINISH_2,
    NUMBER_OF_DAYS_TO_PREDICT,
    MIN_TRAIN_DATE,
    LAGGED_FEATRUES_WINDOW_SIZE,
)


def prepare_data(
    data: pd.DataFrame,
    holidays_events_data: pd.DataFrame,
    oil_data: pd.DataFrame,
    stores_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Collects and prepares raw data.

    Collects raw data by merging it together. Prepares features, renames them.

    Args:
        data: the pd.DataFrame instance with main sales data.
        holidays_events_data: the pd.DataFrame instance with holidays data.
        oil_data: the pd.DataFrame instance with oil prices data.
        stores_data: the pd.DataFrame instance with stores data.

    Returns:
        Tuple(train_data, test_data) where train_data and test_data are prepared
        train and test parts of the data respectively.
    """
    holidays_events_data["priority"] = holidays_events_data["locale"].map(
        {"National": 3, "Regional": 2, "Local": 1}
    )
    holidays_events_data.sort_values(
        by=["date", "priority"], ascending=False, inplace=True
    )
    holidays_events_data.drop_duplicates(subset=["date"], keep="first", inplace=True)
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

    data.set_index("id", inplace=True)
    data["date"] = pd.to_datetime(data["date"])

    data["is_holiday_transferred"] = data["is_holiday_transferred"].map(
        lambda x: False if not x or x == NOT_HOLIDAY_DAY else True
    )

    data.drop(
        [
            "items_on_promotion",
            "oil_price",
            "city",
            "state",
            "store_type",
            "store_cluster",
        ],
        axis=1,
        inplace=True,
    )

    min_test_date = pd.to_datetime(data["date"].unique()[-NUMBER_OF_DAYS_TO_PREDICT])
    train_data = data[pd.to_datetime(data["date"]) < min_test_date]
    test_data = data[pd.to_datetime(data["date"]) >= min_test_date]

    return train_data, test_data


def add_features(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Adds new features to the data.

    Adds new binary, date-related and lagged features depending on the values of
    some other features.

    Args:
        data: pd.DataFrame instance with sales data.

    Returns:
        Tuple(train_data, test_data) where train_data and test_data are train and
        test parts of the data with added new features respectively.
    """
    data = pd.concat([train_data, test_data], axis=0)
    data["date"] = pd.to_datetime(data["date"])
    periods = [
        (OIL_PRICE_FALLING_START_1, OIL_PRICE_FALLING_FINISH_1),
        (OIL_PRICE_FALLING_START_2, OIL_PRICE_FALLING_FINISH_2),
    ]
    data["is_during_oil_prices_falling"] = data["date"].apply(
        lambda date: 1 if any(start <= date <= end for start, end in periods) else 0
    )

    def is_special_unit(unit: Any, special_list: List[Any]) -> int:
        if unit in special_list:
            return 1
        return 0

    data["is_special_non_working_day"] = data["day_type"].apply(
        lambda x: is_special_unit(x, SPECIAL_NON_WORKING_DAYS)
    )
    data["is_popular_holiday"] = data["holiday_status"].apply(
        lambda x: is_special_unit(x, POPULAR_HOLIDAYS)
    )
    data["number_of_days_since_earthquake"] = (
        data["date"] - DATE_OF_EARTHQUAKE
    ).dt.days

    data["days_since_start"] = (pd.to_datetime(data["date"]) - MIN_TRAIN_DATE).dt.days
    data["year"] = pd.to_datetime(data["date"]).dt.year
    data["month"] = pd.to_datetime(data["date"]).dt.month
    data["day"] = pd.to_datetime(data["date"]).dt.day
    data["day_of_week"] = pd.to_datetime(data["date"]).dt.dayofweek

    data.sort_values(by=["store_number", "item_family", "date"], inplace=True)

    min_test_date = pd.to_datetime(data["date"].unique()[-NUMBER_OF_DAYS_TO_PREDICT])
    train_data = data[pd.to_datetime(data["date"]) < min_test_date]
    test_data = data[pd.to_datetime(data["date"]) >= min_test_date]

    train_data[f"mean_sales_last_{LAGGED_FEATRUES_WINDOW_SIZE}_known_days"] = (
        train_data.groupby(["store_number", "item_family"])["item_sales"]
        .transform(
            lambda x: x.shift(1)
            .rolling(window=LAGGED_FEATRUES_WINDOW_SIZE, min_periods=1)
            .mean()
        )
        .bfill()
    )

    last_mean_sales = train_data.loc[
        train_data.groupby(["store_number", "item_family"])["date"].idxmax(),
        [
            "store_number",
            "item_family",
            f"mean_sales_last_{LAGGED_FEATRUES_WINDOW_SIZE}_known_days",
        ],
    ]
    test_data = test_data.merge(
        last_mean_sales, on=["store_number", "item_family"], how="left"
    )

    train_data.drop(["date"], axis=1, inplace=True)
    test_data.drop(["date"], axis=1, inplace=True)

    return train_data, test_data
