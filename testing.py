import tensorflow as tf
import numpy as np
import json
from sklearn.metrics import accuracy_score
import pandas as pd


def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="study_assistant_model.tflite")
    interpreter.allocate_tensors()
    return interpreter


def normalize_input(data, params):
    data_min = np.array(params["min"])
    data_max = np.array(params["max"])
    return (data - data_min) / (data_max - data_min)


def check_accuracy(interpreter, test_data, true_labels, normalization_params):
    normalized_data = normalize_input(test_data, normalization_params)

    predictions = []
    for i in range(len(test_data)):
        input_data = normalized_data[i].reshape(1, -1)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))

        interpreter.invoke()

        prediction = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(prediction)

        predictions.append(predicted_class)
        print(f"True Label: {true_labels[i]}, Predicted Label: {predicted_class}")

    accuracy = accuracy_score(true_labels, predictions)
    return accuracy


def main():
    interpreter = load_tflite_model()
    df = pd.read_csv("test_focus_dataset.csv")
    print(df['Focus Level'].value_counts())

    with open("normalization_params.json", "r") as f:
        normalization_params = json.load(f)

    label_mapping = {"Low": 0, "Medium": 1, "High": 2}
    df["Focus Level"] = df["Focus Level"].map(label_mapping)
    print(df.head())

    #test_data = df[["Heart Rate", "Breathing Rate", "Stress Level"]].values

    # True labels (0 = take a break, 1 = study for some time, 2 = continue studying)
    #true_labels = df["Focus Level"].values
    test_data = np.array([[73, 12, 4]])
    true_labels = np.array([2])

    accuracy = check_accuracy(interpreter, test_data, true_labels, normalization_params)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
