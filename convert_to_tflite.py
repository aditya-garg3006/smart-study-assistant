import tensorflow as tf

# Load the trained Keras model
model = tf.keras.models.load_model("smart_study_assistant_model.h5")

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open("study_assistant_model.tflite", "wb") as f:
    f.write(tflite_model)
