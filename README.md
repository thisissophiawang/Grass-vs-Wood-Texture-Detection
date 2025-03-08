# 🌱🌲 Grass vs. Wood Texture Classification  

![Project Preview](https://github.com/thisissophiawang/Grass-vs-Wood-Texture-Detection/blob/main/texture-classification-preview.svg)

## 📌 Project Overview  
This project implements **two texture classification algorithms** to distinguish between **grass** and **wood** images. The goal is to explore different **texture feature extraction techniques** and evaluate their effectiveness in a machine learning-based classification system.  

### **Objective**  
- Implement and compare **two texture analysis techniques**:  
  - **Gray Level Co-occurrence Matrix (GLCM)**  
  - **Local Binary Patterns (LBP)**  
- Train and evaluate **machine learning classifiers** (e.g., SVM, k-NN, Decision Trees).  
- Develop an **interactive interface** using **Gradio** for real-time texture classification.  
- Analyze the **performance, strengths, and weaknesses** of each approach.  

---

## 🚀 Features  
✅ **Texture Feature Extraction** (GLCM, LBP)  
✅ **Machine Learning Classifiers** (SVM, k-NN, Decision Trees)  
✅ **Data Augmentation** (Rotation, Scaling) for dataset diversity  
✅ **Hyperparameter Tuning** (Cross-validation for model optimization)  
✅ **Interactive Gradio Interface** for real-time classification  

---

## 📂 Dataset Preparation  
- **Image Collection**:  
  - Dataset consists of **100 images** (50 grass, 50 wood).  
  - Ensures **diverse lighting, scale, and texture variations**.  
- **Data Splitting**:  
  - **Training Set**: 70%  
  - **Testing Set**: 30%  

---

## 🛠️ Methodology  

### **1️⃣ Feature Extraction Techniques**  
#### **Technique 1: GLCM (Gray Level Co-occurrence Matrix)**
- Computes **texture properties** such as:  
  - Contrast, correlation, energy, homogeneity  
- Extracts **statistical features** at different angles and distances.  

#### **Technique 2: LBP (Local Binary Patterns)**
- Converts images into **binary patterns** based on local pixel intensity.  
- Generates **LBP histograms** for feature representation.  
- Tested with **different radii and neighbor configurations**.  

---

### **2️⃣ Feature Vector Compilation**
- Each extracted feature is converted into a **numerical vector**.
- Feature vectors are labeled as **"Grass"** or **"Wood"**.
- Stored for **classifier training and evaluation**.

---

### **3️⃣ Classification Models**
- **Support Vector Machine (SVM)**  
- **k-Nearest Neighbors (k-NN)**  
- **Decision Trees**  
- Models trained on both **GLCM and LBP** feature sets.  

---

### **4️⃣ Data Augmentation**
- **Rotation**: 90°, 180°, and 270°  
- **Scaling**: 0.5x and 2x  
- Helps improve model robustness and generalization.  

---

### **5️⃣ Hyperparameter Tuning**
- Used **GridSearchCV with StratifiedKFold cross-validation** for best model selection.

---

## 📊 Results & Evaluation  
- **Performance Metrics**:  
  - Accuracy, Precision, Recall, F1-score  
  - Confusion Matrix, Precision-Recall Curve, ROC Curve  
- **Comparison of GLCM vs. LBP**:  
  - Strengths and weaknesses of each method  
  - Computational efficiency vs. accuracy  

---

## 🖼️ Sample Visualizations  

### **Precision Comparison Between Models**  
![Precision Comparison](https://github.com/thisissophiawang/Grass-vs-Wood-Texture-Detection/blob/main/precision_comparison_image.png?raw=true)  

### **Decision Boundary Visualization**  
![Decision Boundary](https://github.com/thisissophiawang/Grass-vs-Wood-Texture-Detection/blob/main/decision%20broundary.png)  


---

## 🖥️ Interactive Gradio Interface  
### **Features**  
✅ Upload an image (Grass/Wood).  
✅ Choose between **GLCM or LBP** feature extraction.  
✅ View real-time **classification results**.  

### **Run Gradio Interface Locally**
```bash
python visualize_output.py
