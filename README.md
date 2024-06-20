Project aimed at stores' sales prediction

This project uses [this](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) dataset from Kaggle platform.


## Dataset description
The dataset contains info about the sales of Favorita stores located in Ecuador from January 2013 to August 2017. 

## The goal of the problem
The goal is to build a model for making a sales prediction for the following two weeks.

## Models used
We decided to use machine learning models based on the gradient boosting principle since they are known to be pretty efficient and robust.

## Validation methods
The TimeSeriesCrossValidation from Sklearn was chosen for the cross-validation. Its implementation includes the method called Expanding Window and well balanced in terms of computational cost and robustness. Standard Time Series Split, Sliding Window and Nested Cross Validation were among alternatives.

## Metrics 
Since the sales of some items are close to zero or exactly zero, there was no point in using MAPE metric. That's why MAE metric was chosen as well as the mean value of the target variable, so it's rather MAE/Mean as an alternative to MAPE.


## Development
1. Clone this repository to your machine.
2. Download [the dataset](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data), save .csv locally.
3. Make sure Python 3.12.2 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.8.2).
4. Install all requirements (including dev requirements) to poetry environment via your terminal:

```sh
poetry install 
```