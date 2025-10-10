import json, io, warnings, os
warnings.filterwarnings("ignore")

from flask import Flask, request, render_template, jsonify
from PIL import Image
import tensorflow as tf
import numpy as np

MODEL_PATH   = "best_weights_effnetv2_finetuned_adamw.keras"
CLASSES_PATH = "class_names.json"
IMG_SIZE     = 224

with open(CLASSES_PATH, encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

@tf.keras.utils.register_keras_serializable()
class TopKAccFloat32(tf.keras.metrics.TopKCategoricalAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred, tf.float32)
        return super().update_state(y_true, y_pred, sample_weight)

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"TopKAccFloat32": TopKAccFloat32},
    compile=False
)

app = Flask(__name__, template_folder="templates")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image"}), 400
    image = image.resize((IMG_SIZE, IMG_SIZE))

    arr = tf.keras.preprocessing.image.img_to_array(image)
    arr = tf.expand_dims(arr, 0)
    preds = model(arr, training=False)[0]

    idx   = int(tf.argmax(preds))
    label = CLASS_NAMES[idx]
    conf  = float(preds[idx])

    if label == "Other":
        return jsonify({"label": "Not a tree image", "confidence": conf})
    else:
        return jsonify({"label": label, "confidence": conf})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  