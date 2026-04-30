# 🚨 Tourist Safety Anomaly Detection using LSTM Autoencoder

## 📌 Overview

This project builds an **anomaly detection system for tourist movement patterns** using GPS trajectory data.
By analyzing location, speed, and motion behavior over time, the model identifies **unusual or potentially risky movement patterns** that could indicate safety concerns.

The core idea is simple:

> Learn what *normal movement looks like* → flag anything that deviates significantly.

---

## 🎯 Problem Statement

Tourists traveling in unfamiliar environments may face risks such as:

* Getting lost
* Sudden stops in unsafe areas
* Abnormal travel patterns

Traditional rule-based systems fail to capture complex behavioral patterns.
This project uses **deep learning (LSTM Autoencoder)** to detect anomalies in sequential movement data.

---

## 🧠 Approach

### 1. Data Preprocessing

* Loaded GPS trajectory dataset
* Converted timestamps to datetime format
* Sorted data by `track_id` and time
* Handled missing values

---

### 2. Feature Engineering

Key features extracted:

* 📍 Latitude & Longitude
* ⏱ Time elapsed between points
* 📏 Distance (Haversine formula)
* 🚀 Speed (m/s)
* ⚡ Acceleration
* 🧭 Bearing (direction of movement)
* 🕒 Hour of day
* 📅 Day of week

These features capture **spatial + temporal + behavioral dynamics**.

---

### 3. Data Scaling

* Applied **MinMaxScaler** for normalization
* Saved scaler using `joblib` for reuse

---

### 4. Sequence Creation

* Converted data into **time-series sequences**
* Sequence length = `10 timesteps`
* Shape: `(samples, timesteps, features)`

---

### 5. Model Architecture (LSTM Autoencoder)

The model learns to reconstruct normal sequences:

* **Encoder**

  * LSTM (128 units)
  * Compresses sequence into latent representation

* **Decoder**

  * RepeatVector
  * LSTM layers
  * TimeDistributed Dense output

👉 If reconstruction error is high → anomaly

---

### 6. Model Training

* Loss: Mean Absolute Error (MAE)
* Epochs: 50
* Batch size: 32
* Validation split: 10%

---

### 7. Anomaly Detection

* Calculated reconstruction error for each sequence
* Defined threshold:

```
threshold = mean + 2 * std
```

* Any sequence above threshold → **Anomaly**


* learning curve plotted

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* TensorFlow / Keras
* Matplotlib

---


---

## 🚀 How to Run

### 1. Clone repo

```bash
git clone https://github.com/your-username/tourist-safety.git
cd tourist-safety
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib joblib
```

### 3. Run notebook

Open in:

* Jupyter Notebook
* Google Colab

---

## 🔍 Use Cases

* Tourist safety monitoring systems
* Smart city surveillance
* Travel analytics
* Emergency detection systems

---

## 📈 Future Improvements

* Real-time anomaly detection
* Integration with maps (Google Maps API)
* Alert system (SMS / App notification)
* GPS clustering for location risk scoring
* Replace LSTM with Transformer models

---

## 💡 Key Takeaways

* Time-series modeling is powerful for behavioral analysis
* Feature engineering is **critical** for anomaly detection
* Autoencoders work well when labeled anomaly data is scarce

---

## 👨‍💻 Author

**Revanth Kovvuri**

---

## ⭐ If you found this useful

Give the repo a star — it helps a lot!
