from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

MODEL_PATH = "./models/best_model.pkl"

with open(MODEL_PATH, 'rb') as model_file:
    model_pipeline = pickle.load(model_file)

app = FastAPI()

class WaterQualityData(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

@app.get("/")
async def home():
    return {
        "input_example":{
           "ph":10.316400384553162,
           "Hardness":217.2668424334475,
           "Solids":10676.508475429378,
           "Chloramines":3.445514571005745,
           "Sulfate":397.7549459751925,
           "Conductivity":492.20647361771086,
           "Organic_carbon":12.812732207582542,
           "Trihalomethanes":72.28192021570328,
           "Turbidity":3.4073494284238364
        },
        "output_example": {"potabilidad": 0},
    }

@app.post("/potabilidad/")
async def predict_potability(data: WaterQualityData):
    try:
        input_array = np.array([[data.ph, 
                                 data.Hardness, 
                                 data.Solids, 
                                 data.Chloramines,
                                 data.Sulfate, 
                                 data.Conductivity, 
                                 data.Organic_carbon,
                                 data.Trihalomethanes, 
                                 data.Turbidity]])
        prediction = model_pipeline.predict(input_array)
        return {"potabilidad": int(prediction[0])}    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
