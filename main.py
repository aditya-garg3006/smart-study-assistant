from fastapi import FastAPI
import tensorflow.lite as tflite
import numpy as np
from pydantic import BaseModel
import json

app = FastAPI()

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="study_assistant_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("normalization_params.json", "r") as f:
    normalization_params = json.load(f)


# Normalize new input using the saved normalization parameters
def normalize_input(data, params):
    data_min = np.array(params["min"])
    data_max = np.array(params["max"])
    return (data - data_min) / (data_max - data_min)


LABELS = ["low", "medium", "high"]


class InputData(BaseModel):
    heart_rate: float
    breathing_rate: float
    stress_level: float


@app.post("/predict")
async def predict(data: InputData):
    # Prepare input data as NumPy array
    input_data = np.array([[data.heart_rate, data.breathing_rate, data.stress_level]], dtype=np.float32)

    normalized_data = normalize_input(input_data, normalization_params)
    normalized_input = normalized_data[0].reshape(1, -1)

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], normalized_input.astype(np.float32))
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(prediction)

    # (0 for low, 1 for medium, 2 for high)
    # Map to label
    focus_level = LABELS[predicted_class]

    return {"focus_level": focus_level}
