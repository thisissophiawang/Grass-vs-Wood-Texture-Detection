from PIL import Image
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import gradio as gr

# GLCM feature extraction function, set different distances and angles
def extract_glcm_features(image, distances=[1], angles=[0]):
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return [contrast, correlation, energy, homogeneity]

# LBP feature extraction function, set different radius and number of points
def extract_lbp_features(image, radius=1, points=8):
    lbp = local_binary_pattern(image, points, radius, method="uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist

# load images from folder
def load_images_from_folder(folder, label=None):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('L')  # 将图像转换为灰度图像
        img = img.resize((128, 128))  # 调整图像大小
        img_array = np.array(img)
        images.append(img_array)
        if label is not None:
            labels.append(label)
    return images, labels

# load dataset
def load_dataset(root_dir):
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    # Grass data
    grass_train, grass_train_labels = load_images_from_folder(os.path.join(root_dir, 'grass 208 image/train'), 0)
    grass_val, grass_val_labels = load_images_from_folder(os.path.join(root_dir, 'grass 208 image/validation'), 0)
    grass_test, grass_test_labels = load_images_from_folder(os.path.join(root_dir, 'grass 208 image/test'), 0)

    # Wood data
    wood_train, wood_train_labels = load_images_from_folder(os.path.join(root_dir, 'wood 150 image/train'), 1)
    wood_val, wood_val_labels = load_images_from_folder(os.path.join(root_dir, 'wood 150 image/validation'), 1)
    wood_test, wood_test_labels = load_images_from_folder(os.path.join(root_dir, 'wood 150 image/test'), 1)

    # combine the data
    X_train.extend(grass_train + wood_train)
    y_train.extend(grass_train_labels + wood_train_labels)

    X_val.extend(grass_val + wood_val)
    y_val.extend(grass_val_labels + wood_val_labels)

    X_test.extend(grass_test + wood_test)
    y_test.extend(grass_test_labels + wood_test_labels)

    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), np.array(X_test), np.array(y_test)

# Feature extraction (GLCM and LBP)
def extract_features(image, method='GLCM'):
    if method == 'GLCM':
        feature = extract_glcm_features(image)
    elif method == 'LBP':
        feature = extract_lbp_features(image)
    else:
        raise ValueError("Invalid method. Choose between 'GLCM' and 'LBP'.")  
    return np.array(feature).reshape(1, -1)

# Hyperparameter tuning
def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']  # Only relevant for the 'rbf' kernel
    }

    #grid search
    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # return the best model
    return grid_search.best_estimator_

# load dataset
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset('lab2 image')

# Extract GLCM and LBP features for training, validation, and test sets
X_train_glcm = np.vstack([extract_features(img, method='GLCM') for img in X_train])
X_val_glcm = np.vstack([extract_features(img, method='GLCM') for img in X_val])
X_test_glcm = np.vstack([extract_features(img, method='GLCM') for img in X_test])

X_train_lbp = np.vstack([extract_features(img, method='LBP') for img in X_train])
X_val_lbp = np.vstack([extract_features(img, method='LBP') for img in X_val])
X_test_lbp = np.vstack([extract_features(img, method='LBP') for img in X_test])

# Hyperparameter Tuning
best_glcm_model = tune_hyperparameters(X_train_glcm, y_train)
best_lbp_model = tune_hyperparameters(X_train_lbp, y_train)


#verify the accuracy of the model
val_accuracy_glcm = accuracy_score(y_val, best_glcm_model.predict(X_val_glcm))
val_accuracy_lbp = accuracy_score(y_val, best_lbp_model.predict(X_val_lbp))
print(f"Optimized GLCM 验证集准确率: {val_accuracy_glcm}")
print(f"Optimized LBP 验证集准确率: {val_accuracy_lbp}")

# evaluate the model on the test set
y_test_pred_glcm = best_glcm_model.predict(X_test_glcm)
y_test_pred_lbp = best_lbp_model.predict(X_test_lbp)

test_accuracy_glcm = accuracy_score(y_test, y_test_pred_glcm)
test_accuracy_lbp = accuracy_score(y_test, y_test_pred_lbp)
print(f"Optimized GLCM 测试集准确率: {test_accuracy_glcm}")
print(f"Optimized LBP 测试集准确率: {test_accuracy_lbp}")

# print the classification report and confusion matrix
print("---- GLCM Model Evaluation ----")
print("GLCM Accuracy: ", test_accuracy_glcm)
print("GLCM Classification Report:")
print(classification_report(y_test, y_test_pred_glcm, target_names=["Grass", "Wood"]))
print("GLCM Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_glcm))

print("---- LBP Model Evaluation ----")
print("LBP Accuracy: ", test_accuracy_lbp)
print("LBP Classification Report:")
print(classification_report(y_test, y_test_pred_lbp, target_names=["Grass", "Wood"]))
print("LBP Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred_lbp))

# predict function
def predict(image, algorithm):
    image = image.convert('L')  # convert to grayscale
    image = np.array(image.resize((128, 128)))  # resize the image

    # feature extraction
    features = extract_features(image, method=algorithm)
    features = np.array(features).reshape(1, -1)  # convert to 2D array

    # use the best model to predict
    model = best_glcm_model if algorithm == 'GLCM' else best_lbp_model
    prediction = model.predict(features)
    return "Grass" if prediction[0] == 0 else "Wood"

# evaluate_folder function
def evaluate_folder(folder_path, algorithm):
    images, labels = load_images_from_folder(folder_path)
    correct_predictions = 0
    total_images = len(images)

    for img in images:
        pred = predict(Image.fromarray(img), algorithm)
        # grass: 0, wood: 1
        correct_predictions += (pred == "Grass" if labels == 0 else "Wood")

    accuracy = correct_predictions / total_images
    return f"{algorithm} The accuracy of the model in the folder is: {accuracy:.2f}"

# Gradio interface
iface = gr.Interface(
    fn=predict,  # predict function
    inputs=[gr.Image(type='pil', label="Upload Image"),
            gr.Dropdown(choices=["GLCM", "LBP"], label="Choose Algorithm")],
    outputs="text",
    live=True,
    title="Texture Classification",
    description="Upload an image and select an algorithm (GLCM or LBP) to classify the texture as Grass or Wood."
)

# folder interface
folder_iface = gr.Interface(
    fn=evaluate_folder,
    inputs=[gr.Textbox(label="input folder path"),
            gr.Dropdown(choices=["GLCM", "LBP"], label="pick an algorithm")],
    outputs="text",
    title="evaluate folder",
)

# gradio interface
iface.launch()
folder_iface.launch()
