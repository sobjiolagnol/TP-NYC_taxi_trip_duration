
from fastapi import FastAPI
from pydantic import BaseModel
import requests
import random
import requests

class TaxiTrip(BaseModel):
    pickup_datetime: str
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    passenger_count: int

app = FastAPI()

@app.post("/predict")
def predict_duration(trip: TaxiTrip):
    duration = random.randint(1, 60)
    return {"predicted_duration": duration}
