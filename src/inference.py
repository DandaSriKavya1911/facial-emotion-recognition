# src/inference.py
import json
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

DEFAULT_MODEL_PATH = "models/emotion_recognition_model.h5"
DEFAULT_CLASSMAP_PATH = "models/class_indices.json"

def load_class_map(classmap_path=DEFAULT_CLASSMAP_PATH):
    with open(classmap_path, "r") as f:
        class_indices = json.load(f)
    # invert mapping to get labels by index
    idx2label = {v:k for k,v in class_indices.items()}
    # ensure order by index
    labels = [idx2label[i] for i in range(len(idx2label))]
    return labels

def predict_emotion_on_image(image_path, model_path=DEFAULT_MODEL_PATH, classmap_path=DEFAULT_CLASSMAP_PATH, show=False):
    model = load_model(model_path)
    labels = load_class_map(classmap_path)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    results = []
    if len(faces) == 0:
        return results

    for (x,y,w,h) in faces:
        roi_color = img_rgb[y:y+h, x:x+w]
        roi = cv2.resize(roi_color, (224,224))
        roi = roi.astype("float32") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi)[0]
        idx = int(np.argmax(preds))
        label = labels[idx]
        confidence = float(np.max(preds))
        results.append({"box":(x,y,w,h), "label":label, "confidence":confidence})

        if show:
            import matplotlib.pyplot as plt
            # draw bounding box on original BGR image for display
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img, f"{label} ({confidence*100:.1f}%)", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36,255,12), 2)

    if show:
        import matplotlib.pyplot as plt
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis("off"); plt.show()

    return results
