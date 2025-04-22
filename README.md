
# 🧠 Smart Study Assistant

The **Smart Study Assistant** is an IoT + AI-based project that helps monitor and enhance a student’s focus during study sessions. It does so by analyzing physiological parameters in real time using an ESP32-based simulation and an ML model deployed via a FastAPI backend.

---

## 🚀 Project Overview

This system predicts a user's **focus level**—categorized as **High**, **Medium**, or **Low**—based on three physiological signals:

- ❤️ Heart Rate  
- 🌬️ Breathing Rate  
- ⚡ Stress Level

---

## 🧠 How It Works

1. **Model Training (`model.py`)**  
   Trains and saves a classification model using sample physiological data.

2. **FastAPI Backend (`main.py`)**  
   Serves the trained model through an API endpoint. Hosted using [Render](https://render.com/).

3. **Simulation via Wokwi**  
   An [ESP32 microcontroller](https://www.espressif.com/en/products/socs/esp32) is used to send simulated data to the FastAPI backend deployed on Render.

---

## 🌐 Live Demo Links

- **API (FastAPI on Render):**  
  🔗 [https://smart-study-assistant.onrender.com](https://smart-study-assistant.onrender.com)

- **ESP32 Simulation (Wokwi):**  
  🔗 [https://wokwi.com/projects/421228844698976257](https://wokwi.com/projects/421228844698976257)

---

## ⚠️ Important Note: First-time Use

Render apps go to sleep when idle. So the **first call** to the API (especially from Wokwi) may fail as it takes time to load dependencies like **TensorFlow**.

### ✅ Recommended Steps:

1. **Before using Wokwi**, make a request to the API endpoint using:
   - [Postman](https://www.postman.com/)  
   - or [Hoppscotch](https://hoppscotch.io/)

2. Use the POST request given at the end to activate the app.

Once the server is "warmed up", the Wokwi simulation should work smoothly.

---

## 🧰 Tech Stack

- **Machine Learning:** TensorFlow, Scikit-learn  
- **Backend:** Python, FastAPI  
- **Simulation:** Wokwi (ESP32)  
- **Deployment:** Render

---

## 📬 API Endpoint

**POST** `/predict`

**Payload format:**

```json
{
  "heart_rate": 85,
  "breathing_rate": 18,
  "stress_level": 3
}
```

**Response:**

```json
{
  "focus_level": "Medium"
}
```

---
