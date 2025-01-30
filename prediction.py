import tensorflow as tf
import numpy as np
import json


def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="study_assistant_model.tflite")
    interpreter.allocate_tensors()
    return interpreter


# Load the normalization parameters (min and max values) from the saved file
with open("normalization_params.json", "r") as f:
    normalization_params = json.load(f)


# Normalize new input using the saved normalization parameters
def normalize_input(data, params):
    data_min = np.array(params["min"])
    data_max = np.array(params["max"])
    return (data - data_min) / (data_max - data_min)


# Prediction function
def predict_focus(interpreter, new_input):
    normalized_data = normalize_input(new_input, normalization_params)
    normalized_input = normalized_data[0].reshape(1, -1)

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor with normalized data
    interpreter.set_tensor(input_details[0]['index'], normalized_input.astype(np.float32))

    # Invoke the model (run inference)
    interpreter.invoke()

    # Get the prediction (output tensor)
    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(prediction)

    # Return the class with the highest probability (0 for low, 1 for medium, 2 for high)
    return predicted_class
