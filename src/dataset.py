# src/dataset.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(
    train_dir,
    img_size=(224,224),
    batch_size=32,
    validation_split=0.2,
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    seed=42
):
    datagen = ImageDataGenerator(
        rescale=rescale,
        validation_split=validation_split,
        rotation_range=rotation_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip
    )

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=seed
    )

    val_generator = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=True,
        seed=seed
    )

    return train_generator, val_generator
