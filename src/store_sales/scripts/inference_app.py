from fastapi import FastAPI
from pydantic import BaseModel
from store_sales.modules import MODELS_FOLDER_PATH
import pickle
import pandas as pd
import numpy as np


class InferenceRequest(BaseModel):
    store_number: int
    item_family: str
    day_type: str
    holiday_status: str
    holiday_location: str
    holiday_description: str
    is_holiday_transferred: bool
    mean_sales_last_30_known_days: float
    is_during_oil_prices_falling: int
    is_special_non_working_day: int
    is_popular_holiday: int
    number_of_days_since_earthquake: int
    days_since_start: int
    year: int
    month: int
    day: int
    day_of_week: int

app = FastAPI()

def get_model(store_number: int, item_family: str):
    models_file = MODELS_FOLDER_PATH / "models.pkl"
    with open(models_file, "rb") as f:
        models = pickle.load(f)
    model = models[(store_number, item_family)]
    return model

@app.post("/predict")
def get_predict(request: InferenceRequest):
    model = get_model(request.store_number, request.item_family)
    
    instance_to_predict = pd.DataFrame([{
        "store_number": request.store_number,
        "item_family": request.item_family,
        "day_type": request.day_type,
        "holiday_status": request.holiday_status,
        "holiday_location": request.holiday_location,
        "holiday_description": request.holiday_description,
        "is_holiday_transferred": request.is_holiday_transferred,
        "mean_sales_last_30_known_days": request.mean_sales_last_30_known_days,
        "is_during_oil_prices_falling": request.is_during_oil_prices_falling,
        "is_special_non_working_day": request.is_special_non_working_day,
        "is_popular_holiday": request.is_popular_holiday,
        "number_of_days_since_earthquake": request.number_of_days_since_earthquake,
        "days_since_start": request.days_since_start,
        "year": request.year,
        "month": request.month,
        "day": request.day,
        "day_of_week": request.day_of_week
    }])


    cat_columns = [col for col in instance_to_predict.columns if instance_to_predict[col].dtype == "object"]
    for col in cat_columns:
        instance_to_predict[col] = instance_to_predict[col].astype("category")
        
    prediction = model.predict(instance_to_predict)
    
    return {"prediction": np.round(prediction[0], 2)}

