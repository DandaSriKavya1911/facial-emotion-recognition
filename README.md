# Facial Emotion Recognition 😃

A **Facial Emotion Recognition (FER)** system built using **VGG16 + LSTM** to classify human facial expressions into seven emotion categories — _Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral_ — using the **FER2013 dataset**.

---

## 📂 Project Overview

This project detects human emotions from images or real-time webcam feeds using a deep learning model.  
It demonstrates **end-to-end machine learning workflow**: preprocessing, model training, evaluation, and prediction.

---

## 🧩 Features

- Image preprocessing and data augmentation
- Transfer Learning using **VGG16**
- Sequential modeling with **LSTM**
- Emotion classification (7 classes)
- Evaluation with **Confusion Matrix & Accuracy Metrics**
- Simple **inference script (`predict.py`)** for single-image prediction
- Clean, modular code structure (`src/` folder)
- Colab-ready for quick demo
- MIT Licensed (open source)

---

## 🧠 Model Architecture

1. **VGG16** pretrained on ImageNet used as a feature extractor
2. **LSTM** network added to capture temporal/spatial emotion patterns
3. **Softmax output layer** with 7 neurons representing emotion categories

---

## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas, OpenCV, Matplotlib, Seaborn
- Scikit-learn for metrics
- Gradio (optional) for demo app interface

---

## 🧾 Dataset

**FER2013 Dataset**

- Source: [Kaggle - Facial Expression Recognition Challenge](https://www.kaggle.com/datasets/msambare/fer2013)
