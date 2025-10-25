# 🌿 Tree Species Classification & Web App 🌳

This project is a deep learning-based web application that classifies tree species based on uploaded or linked leaf images. It was built as part of a machine learning exploration to compare multiple CNN architectures and demonstrate a production-ready deployment using a simple UI.

---

## 📌 Project Overview

- ✅ Compared 6 different CNN variants including:
  - Basic CNN (from scratch)
  - MobileNetV2 (head-only and fine-tuned)
  - EfficientNetV2-B0 (head-only and fine-tuned with AdamW)
- 📈 Metrics used: Accuracy, Top-3 Accuracy, Confusion Matrix, RMSE, MAE, R² Score
- 🚫 Integrated Smart Rejection: Labels images as "Not a tree image" if confidence is low or belongs to "Other" class

---

## 🌐 Web App Features

Built using **Flask + Tailwind CSS**, the frontend provides a responsive, mobile-friendly interface with the following features:

- 🔍 Upload or drag & drop image
- 🧠 Real-time prediction from EfficientNetV2 model
- 📸 Live thumbnail preview
- 📊 Top-3 species predictions with confidence scores
- ❌ Rejection logic for invalid images
- 🔁 Clear/Reset form functionality

---
---
## 📈 Model Performance

| Model | Params | Validation Accuracy | Macro-F1 | Top-3 Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| CNN (Scratch) | 1.2 M | 26.9 % | 0.24 | — |
| MobileNetV2 (head-only) | 2.2 M | 68.4 % | 0.68 | 87 % |
| MobileNetV2 (fine-tuned, SGD) | 2.4 M | 69.7 % | 0.68 | 89 % |
| MobileNetV2 (fine-tuned, AdamW) | 2.4 M | 72.1 % | 0.68 | 90 % |
| EfficientNetV2-B0 (head-only) | 4.2 M | 76.5 % | 0.70 | 94 % |
| EfficientNetV2-B0 (fine-tuned) | 4.2 M | 77.2 % | 0.70 | 96 % |
---
## 📁 Directory Structure
```
Tree-species-classification/
├── dataset/                    # Training and validation data for the models
├── models/                     # Pre-trained models and checkpoints
├── tree_app/
│   ├── app.py                  # Flask backend
│   ├── templates/
│   │   └── index.html          # Frontend UI
│   ├── static/                 # Assets (CSS, JS, images, etc.)
│   ├── class_names.json        # Tree species label mapping
│   ├── best_weights_effnetv2_finetuned_adamw.keras
│   └── README.md
└── .gitignore                  # Git ignore file to exclude unnecessary files
```



---
---

## 📸 Screenshots

| Sucessfull Classification| Rejection System|
| :---: | :---: |
|The web application accurately identified **'Mountain Ebony'** with high confidence, demonstrating precise classification capabilities.<img width="545" height="856" alt="image" src="https://github.com/user-attachments/assets/38b1382e-fc8b-48ea-b2d9-1bb9d7632ac5" />|The system correctly rejected an unrelated input, classifying it as **"Not a tree image,"** highlighting the robust **"Other"** class and rejection threshold.<img width="540" height="845" alt="image" src="https://github.com/user-attachments/assets/08e2abdc-4dc5-4502-a7fd-60d97d61c7ab" />|

---
## 💻 How to Run Locally

### 1. Clone the repository

```
git clone https://github.com/your-username/tree-species-classifier.git
cd tree-species-classifier
```
### 2. Install dependencies
```
pip install -r requirements.txt
**Or install manually:**
pip install flask tensorflow pillow numpy matplotlib seaborn scikit-learn tqdm
```
### 3. Start the server**
```
python app.py
Go to http://localhost:5000 to use the app.
```
## 👨‍💻 Author
**Yash Agrawal**
* B.Tech (CST), UEM Kolkata
* Email: yash.agrawal0303@gmail.com
* Passionate about AI, web development, and real-world problem solving.

## 📃 License
MIT License – feel free to use, modify, or contribute.

## ✨ Acknowledgments
* TensorFlow/Keras
* Hugging Face (for model ideas)
* Tailwind CSS for frontend styling
* UEM, Kolkata – for academic inspiration
