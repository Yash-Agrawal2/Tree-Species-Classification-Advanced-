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

## 📁 Directory Structure

tree_app/
├── app.py # Flask backend
├── templates/
│ └── index.html # Frontend UI
├── static/ # Assets (optional)
├── class_names.json # Tree species label mapping
├── best_weights_effnetv2_finetuned_adamw.keras
└── README.md


---

## 💻 How to Run Locally

### 1. Clone the repository

```
git clone https://github.com/your-username/tree-species-classifier.git
cd tree-species-classifier
```
###2. Install dependencies
```
pip install -r requirements.txt
**Or install manually:**
pip install flask tensorflow pillow numpy matplotlib seaborn scikit-learn tqdm
```
###3. Start the server**
```
python app.py
Go to http://localhost:5000 to use the app.
```
---
##👨‍💻 Author
Manish Kumar Gupta
B.Tech (CST), UEM Kolkata
Email: mkrock2397456@gmail.com
Passionate about AI, web development, and real-world problem solving.

##📃 License
MIT License – feel free to use, modify, or contribute.

##✨ Acknowledgments

-TensorFlow/Keras
-Hugging Face (for model ideas)
-Tailwind CSS for frontend styling

UEM, Kolkata – for academic inspiration
---
