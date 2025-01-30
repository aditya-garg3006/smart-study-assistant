import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import json

# Step 1: Load and Prepare Data
# Example dataset (replace this with your actual data)

df = pd.read_csv("focus_dataset.csv")

# Convert categorical labels to numerical values
label_mapping = {"Low": 0, "Medium": 1, "High": 2}
df["Focus Level"] = df["Focus Level"].map(label_mapping)

# Split features and target
X = df[["Heart Rate", "Breathing Rate", "Stress Level"]].values
y = df["Focus Level"].values

# Step 2: Normalize Data
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Save the normalization parameters for later use
normalization_params = {
    "min": scaler.data_min_.tolist(),
    "max": scaler.data_max_.tolist(),
}
with open("normalization_params.json", "w") as f:
    json.dump(normalization_params, f)

# Step 3: Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Step 4: Define and Train the Neural Network Model
model = Sequential([
    Dense(32, activation="relu", input_shape=(3,)),  # Input layer
    Dense(16, activation="relu"),                    # Hidden layer
    Dense(3, activation="softmax")                   # Output layer
])


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=50, batch_size=4, validation_data=(X_test, y_test))

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss * 100:.2f}%")

# Step 5: Save the Model
model.save("smart_study_assistant_model.h5")
print("Model saved successfully!")
