import pandas as pd
import numpy as np

# Parameters
num_samples = 1000  # Number of rows in the dataset
np.random.seed(42)  # For reproducibility


# Generate synthetic data
def generate_dataset(num_samples):
    data = []
    for _ in range(num_samples):
        # Generate random values
        heart_rate = np.random.randint(50, 110)  # 50-110 bpm
        breathing_rate = np.random.randint(10, 25)  # 10-25 breaths/min
        stress_level = np.random.randint(1, 11)  # 1-10

        # Check how many values are outside the normal range
        outside_normal = 0
        if not (60 <= heart_rate <= 100):
            outside_normal += 1
        if not (12 <= breathing_rate <= 20):
            outside_normal += 1
        if not (1 <= stress_level <= 6):
            outside_normal += 1

        # Assign focus level
        if outside_normal == 0:
            focus_level = "High"
        elif outside_normal == 1:
            focus_level = "Medium"
        else:
            focus_level = "Low"

        # Append to dataset
        data.append([heart_rate, breathing_rate, stress_level, focus_level])

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=["Heart Rate", "Breathing Rate", "Stress Level", "Focus Level"])
    return df


# Generate dataset
dataset = generate_dataset(num_samples)

# Save to CSV
dataset.to_csv("test_focus_dataset.csv", index=False)

print("Dataset generated and saved as 'focus_dataset.csv'")
print(dataset.head())