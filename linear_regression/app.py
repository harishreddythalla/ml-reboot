from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import json

with open("model/linear_model.json","r") as f:
    weights = json.load(f)

m = weights["m"]
c = weights["c"]


app = FastAPI()

class InputData(BaseModel):
    x: float

@app.post("/predict")
def predict(data: InputData):
    x = data.x
    y_pred = m * x + c
    return {"prediction" : y_pred}
