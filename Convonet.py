from tensorflow.keras.applications.convnext import preprocess_input as convnext_prep
import os, shutil, hashlib, pickle, gzip, json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.notebook import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2, EfficientNetV2S
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
DATA_DIR = '/content/drive/MyDrive/Tree_Species_Dataset'
print(os.listdir(DATA_DIR)[:10])
BATCH = 32
IMG_SIZE = 224
SEED = 42


#  Class Weights 
y_int = train_gen.labels
class_weights = dict(enumerate(
    compute_class_weight('balanced', classes=np.unique(y_int), y=y_int)
))

#  Human-readable class names 
class_names = list(train_gen.class_indices.keys())
readable = [english_names.get(name.replace('_', ' '), name) for name in class_names]
print("Readable class names:", readable)


# 1. Build the generator **with** the preprocessing function
aug_conv = ImageDataGenerator(
    preprocessing_function=convnext_prep,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# 2. Now create the iterators
train_gen_conv = aug_conv.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=BATCH,
    class_mode='categorical',
    subset='training',
    seed=SEED
)

val_gen_conv = aug_conv.flow_from_directory(
    DATA_DIR,
    target_size=(224, 224),
    batch_size=BATCH,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=SEED
)

from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras import layers, models, regularizers

def build_convnext_tiny(num_classes, input_shape=(224, 224, 3), dropout_rate=0.3):
    base = ConvNeXtTiny(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base.trainable = False         
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='gelu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=base.input, outputs=outputs)