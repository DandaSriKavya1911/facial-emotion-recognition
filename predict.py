#!/usr/bin/env python3
# predict.py
import argparse
from src.inference import predict_emotion_on_image

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--model", default="models/emotion_recognition_model.h5", help="Path to model file")
    p.add_argument("--classmap", default="models/class_indices.json", help="Path to class indices JSON")
    p.add_argument("--show", action="store_true", help="Show image with bounding box")
    args = p.parse_args()

    results = predict_emotion_on_image(args.image, model_path=args.model, classmap_path=args.classmap, show=args.show)
    if not results:
        print("No faces detected.")
    else:
        for r in results:
            print(f"{r['label']} ({r['confidence']*100:.2f}%)  box={r['box']}")
