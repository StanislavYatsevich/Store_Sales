Project aimed at stores' sales prediction

This project uses [this](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) dataset from Kaggle platform.


## Dataset description
The dataset contains the information about the daily sales of Favorita stores located in Ecuador per day from January 2013 to August 2017. The whole data consists of several .csv files.

– The train.csv file gives the info about the sales of a certain family of products('family') in a certain store('store_nbr') on a certain day('date') where the target variable 'sales' stands for the number of items(or number of kilos since some items are sold in fractional units, e.g. 2.5 kilos of cheese) of this family sold on this day. Variable 'onpromotion' gives the total number of items in a product family that were being promoted at a store at a given date. This file contains the info from 1st January, 2013 to 15th August, 2017.

– The test.csv file has the same info as train.csv except for the target variable 'sales'. The main challenge is to predict this variable on there days. This file contains the info from 16th to 31st August, 2017.

– The oil.csv file contains the information about the oil prices for the given period.

– The holiday_events.csv file represents the info about different holidays and events in Ecuador for the given period with its characterictics(type, status, location, name, was it transferred to another day or not).

– The stores.csv file contains the info about the stores(number, city, state, type, cluster).

– The sample_submission.csv file is the example of the file needed to be loaded to Kaggle for getting the results.

– The transactions.csv file gives the info about the number of transactions done in a certain store on a given day. However, there's no such info for the test part of the dataset, so we won't use this file.


## The goal of the problem
The goal is to build a model for making a prediction of the target variable 'sales' for two weeks from 16th to 31st August, 2017.


## Models used
We decided to use machine learning models based on the gradient boosting principle since they are known to be pretty efficient and robust.


## Multiple time series technique
Splitting the data by many distinct time series for every unique combination store-item and making an individual prediction for each of them was empirically found to be much better than a staightforward prediction without such splitting.


## Validation methods
The [TimeSeriesSplit() from Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) with n_splits=5 was chosen for performing the cross-validation. Its implementation includes the method called [Expanding Window](https://forecastegy.com/posts/time-series-cross-validation-python/#simple-time-split-validation) and well balanced in terms of computational cost and robustness. Standard Time Series Split, Sliding Window and Nested Cross Validation were among alternatives.


## Metrics 
– Since the sales of some items are close to zero or exactly zero, there was no point in using a default MAPE metric for getting a percentage error.
– That's why WMAPE(Weighted Mean Absolute Percentage Error) was chosen as the main one since it's much less sensitive to zero values of the target variable.
![The WMAPE formula:](https://miro.medium.com/v2/resize:fit:440/1*L358vwYHsmqT5Sqzrs-arA.png)
– The final value of the metric is calculated as mean among all values across all folds in all distict time series.


## Development
1. Clone this repository to your machine.
2. Download [the dataset](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data), save .csv locally. Create a folder path data/raw_data in the directory with the project and save .csv files there.
3. Make sure Python 3.12.3 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.8.2).
4. Install all requirements (including dev requirements) to poetry environment via your terminal:

```sh
poetry install 
```