# src/train.py
import os
import json
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from src.dataset import create_generators
from src.model import build_vgg16_classifier

def train(
    train_dir,
    output_model_path="models/emotion_recognition_model.h5",
    output_classmap_path="models/class_indices.json",
    img_size=(224,224),
    batch_size=32,
    epochs=15
):
    os.makedirs(os.path.dirname(output_model_path) or ".", exist_ok=True)

    train_gen, val_gen = create_generators(train_dir, img_size=img_size, batch_size=batch_size)

    num_classes = train_gen.num_classes
    model = build_vgg16_classifier(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes, freeze_until_last_n=4)

    # callbacks
    checkpoint = ModelCheckpoint(output_model_path, monitor="val_accuracy", save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, verbose=1)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    # Save class_indices mapping
    class_indices = train_gen.class_indices  # e.g. {'angry': 0, 'happy': ...}
    with open(output_classmap_path, "w") as f:
        json.dump(class_indices, f)

    # Optional: save training plots
    plot_history(history, out_path="docs/training_history.png")

    return model, history, class_indices

def plot_history(history, out_path="docs/training_history.png"):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title("Accuracy")
    plt.subplot(1,2,2)
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path)
    plt.close()
