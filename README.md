# Dummy Data Test: Sensor Pipeline Simulation

This repository contains a simulated data pipeline to test wearable sensor data preprocessing and segmentation for human activity recognition (HAR) or health monitoring tasks.

## 📌 Project Overview

We simulate:
- Heart rate data (normally distributed)
- 3-axis accelerometer signals (X, Y, Z)
- Activity labels (walking, standing, sitting)

The data is formatted as a time series, then processed using a **sliding window approach** for segmentation and **label encoding** for future classification tasks.

## 🛠 Features

- Simulates 300 seconds of data with 0.5-second intervals
- Generates data with randomized sensor values and class labels
- Prepares data for ML model input using overlapping sliding windows
- Encodes activity labels using `LabelEncoder`
- Provides visualization of a sample segment

## 🧪 How to Use

### ▶️ Run with Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/StellaOvO-creator/dummy-data-test/blob/main/dummy_sensor_pipeline_en.ipynb)

### 📥 Run Locally

```bash
pip install numpy pandas matplotlib scikit-learn
python dummy_sensor_pipeline_en.py
```

## 📊 Example Output

```text
Original Data Sample:
   time  heart_rate     acc_x     acc_y     acc_z activity
0   0.0   77.483571  0.357787  0.812525 -1.594428  sitting
...

Number of segments: 56
Encoded labels: ['walking', 'sitting', 'standing']
```

## 📁 Files

| File                             | Description                          |
|----------------------------------|--------------------------------------|
| `dummy_sensor_pipeline_en.py`    | Main script (with English comments)  |
| `dummy_sensor_pipeline_en.ipynb` | Jupyter Notebook version             |
| `README.md`                      | This file                            |

## 📌 Author

Created by Jingyu Tang and team as part of COMP5703 IoT-ML Capstone Project.

---

