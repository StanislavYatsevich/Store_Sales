Project aimed at stores' sales prediction

The project uses [the dataset](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) from Kaggle platform.uj


## Dataset description
The dataset contains the information about the daily sales of Favorita stores located in Ecuador per day from January 2013 to August 2017. The whole data consists of several .csv files.

– The train.csv file gives the info about the sales of a certain family of products('family') in a certain store('store_nbr') on a certain day('date') where the target variable 'sales' stands for the number of item (or number of kilos since some items are sold in fractional units, e.g. 2.5 kilos of cheese) of this family sold on this day. Variable 'onpromotion' gives the total number of items in a product family that were being promoted at a store at a given date. This file contains the info from 1st January, 2013 to 15th August, 2017.

– The test.csv file has the same info as train.csv except for the target variable 'sales'. The main challenge is to predict this variable on there days. This file contains the info from 16th to 31st August, 2017.

– The oil.csv file contains the information about the oil prices for the given period.

– The holiday_events.csv file represents the info about different holidays and events in Ecuador for the given period with its characterictics(type, status, location, name, was it transferred to another day or not).

– The stores.csv file contains the info about the stores(number, city, state, type, cluster).

– The sample_submission.csv file is the example of the file needed to be loaded to Kaggle for getting the results.

– The transactions.csv file gives the info about the number of transactions done in a certain store on a given day. However, there's no such info for the test part of the dataset, so we won't use this file.


## The goal of the problem
The goal is to build a model for making a prediction of the target variable 'sales' for two weeks from 16th to 31st August, 2017. That actually means that we need to make a separate prediction of the variable 'sales' for every item family in every store for every day during the given period.


## Multiple time series technique
Splitting the data by many distinct time series for every unique combination store-item and making an individual prediction for each of them was empirically found to be much better than a staightforward prediction without such splitting.


## Validation methods
The [TimeSeriesSplit() from Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) with n_splits=5 was chosen for performing the cross-validation. It means that there will be 5 distinct splits for each time series with the fixed size of the test part equal to 278 days. Its implementation includes the method called [Expanding Window](https://forecastegy.com/posts/time-series-cross-validation-python/#simple-time-split-validation) and well balanced in terms of computational cost and robustness. Standard Time Series Split, Sliding Window and Nested Cross Validation were among alternatives.


## Metrics 
– Since the sales of some items are close to zero or exactly zero, there was no point in using a default MAPE metric for getting a percentage error.

– That's why WMAPE (Weighted Mean Absolute Percentage Error) was chosen as the main one since it's much less sensitive to zero values of the target variable.

![The WMAPE formula:](https://miro.medium.com/v2/resize:fit:440/1*L358vwYHsmqT5Sqzrs-arA.png)

– The final value of the metric is calculated as mean MAE (Mean Absolute Error) across all splits in all distict time series divided by the mean value of the target variable.


## Modeling
– We decided to use models based on the [gradient boosting principle](https://en.wikipedia.org/wiki/Gradient_boosting). It a nutshell, it means that we have multiple consistent models making a prediction based on the previous models' mistakes so that the metric's value is optimized per iteration
– One of the models we used is the [XBGRegressor from XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html).
– For this model we used [Optuna library](https://optuna.org/) for hyperparameter optimization. We set a range of values for each hyperparameter and found optimal ones with help of Optuna tools.
– We decided to choose a random subset of 5 different shops for this task since it would have taken too much time otherwise.
– As a result we managed to improve our metric (29.3% WMAPE before optimization and 26.9% after it). 


## Streamlit dashboard
While performing the Exploratory Data Analysis we created some plots for finding out some possible implicit dependencies in the data for building a better model. We decided to create an interactive dashboard using the [Streamlit tools](https://streamlit.io/) for a more convenient way to explore the data where you can decide which plot to view and which its categories to choose. Navigate to the folder with .py files (as for me, they are in src/store_sales folder in the main directory of the project):

```sh
cd src/store_sales
```

And then run in your terminal:

```sh
streamlit run streamlit_app.py 
```

Then there will be your Local URL (probably http://localhost:8501). Copy it and paste to your browser and enjoy using the dashboard.
The path to the file used for creating the dashboard is set in constants.py file (DATA_FOR_STREAMLIT_PATH variable).


## Development
1. Clone this repository to your machine.
2. Download [the dataset](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data). Create a folder path data/raw_data in the directory with the project and save all .csv files there.
3. Make sure Python 3.12.3 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.8.2).
4. Install all requirements (including dev requirements) to poetry environment via your terminal:

```sh
poetry install 
```

5. Run the splitting_and_preparing_data.py script for splitting the data and preparing it separately so that it's guaranteed there won't be data leakage from future to past. There's a click command line interface implemented so that you can set the paths to input and output folders manually. First navigate to the folder with .py files as it was described in the section about Streamlit. Then run in your terminal:

```sh
python splitting_and_preparing_data.py --raw_data_folder_path <path to the folder with raw data> --prepared_data_folder_path <path to the output folder> 
```

The default values of these paths are set in constants.py file (RAW_DATA_FOLDER_PATH and PREPARED_DATA_STAGE_1_FOLDER_PATH variables respectively).

6. Then run the adding_features.py script for adding certain new features (which might be useful according to the Exploratory Data Analysis). There's also a click command line interface. Same, at first navigate to the folder with .py files as it was described in the section about Streamlit. Then run in your terminal:

```sh
python adding_features.py --input_data_folder_path <path to the folder with input data> --processed_data_folder_path <path to the output folder> 
```

The default values of these paths are also set in constants.py file (PREPARED_DATA_STAGE_1_FOLDER_PATH and PREPARED_DATA_STAGE_2_FOLDER_PATH variables respectively). Pay attention to the fact that the value of the --input_data_folder_path parameter MUST BE THE SAME as the value of the --prepared_data_folder_path parameter from the 5th point since data preparation and feature engineering are performed sequentially in two stages. That's why the default value of these variables are same and refer to a single constant from constants.py.


7. Some libraries (Ruff, Black) for effective code usage and formatting were also used. For using them run in your terminal:

```sh
ruff check <path to the folder with your .py files>
```

```sh
black <path to the folder with your .py files>
```