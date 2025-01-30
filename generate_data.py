import pandas as pd
import numpy as np

num_samples = 1000
np.random.seed(42)


def generate_dataset(num_samples):
    data = []
    for _ in range(num_samples):
        heart_rate = np.random.randint(50, 110)
        breathing_rate = np.random.randint(10, 25)
        stress_level = np.random.randint(1, 11)

        outside_normal = 0
        if not (60 <= heart_rate <= 100):
            outside_normal += 1
        if not (12 <= breathing_rate <= 20):
            outside_normal += 1
        if not (1 <= stress_level <= 6):
            outside_normal += 1

        if outside_normal == 0:
            focus_level = "High"
        elif outside_normal == 1:
            focus_level = "Medium"
        else:
            focus_level = "Low"

        data.append([heart_rate, breathing_rate, stress_level, focus_level])

    df = pd.DataFrame(data, columns=["Heart Rate", "Breathing Rate", "Stress Level", "Focus Level"])
    return df


dataset = generate_dataset(num_samples)
dataset.to_csv("test_focus_dataset.csv", index=False)
print("Dataset generated and saved as 'focus_dataset.csv'")
print(dataset.head())