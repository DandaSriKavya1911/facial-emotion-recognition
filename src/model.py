# src/model.py
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, Model

def build_vgg16_classifier(input_shape=(224,224,3), num_classes=7, freeze_until_last_n=4):
    """
    Build VGG16-based classifier as in training.ipynb.
    freeze_until_last_n: number of final layers to keep trainable
    """
    base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    # freeze all except last `freeze_until_last_n` layers
    if freeze_until_last_n is not None:
        for layer in base.layers[:-freeze_until_last_n]:
            layer.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
