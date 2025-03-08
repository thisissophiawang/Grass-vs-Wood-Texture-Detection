from PIL import Image
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
import gradio as gr

# Optimized GLCM feature extraction function
def extract_glcm_features(image, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    features = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in properties:
        features.append(graycoprops(glcm, prop).flatten())
    return np.hstack(features)

# Optimized multi-scale LBP feature extraction function
def extract_multiscale_lbp_features(image):
    radii = [1, 2, 3]
    n_points = [8, 16, 24]
    features = []
    for radius, points in zip(radii, n_points):
        lbp = local_binary_pattern(image, points, radius, method="uniform")
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        features.append(hist)
    return np.hstack(features)

# Load images from folder
def load_images_from_folder(folder, label=None):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = img.resize((128, 128))  # Resize image
        img_array = np.array(img)
        images.append(img_array)
        if label is not None:
            labels.append(label)
    return images, labels

# Load entire dataset (train, validation, and test sets)
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

    # Combine grass and wood data
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
        feature = extract_multiscale_lbp_features(image)
    else:
        raise ValueError("Invalid method: choose 'GLCM' or 'LBP'")
    return np.array(feature).reshape(1, -1)

# Hyperparameter tuning
def tune_hyperparameters(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    }

    grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

# Main function
def main():
    # Load dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset('lab2 image')

    # Extract GLCM and multi-scale LBP features for train, validation, and test sets
    X_train_glcm = np.vstack([extract_features(img, method='GLCM') for img in X_train])
    X_val_glcm = np.vstack([extract_features(img, method='GLCM') for img in X_val])
    X_test_glcm = np.vstack([extract_features(img, method='GLCM') for img in X_test])

    X_train_lbp = np.vstack([extract_features(img, method='LBP') for img in X_train])
    X_val_lbp = np.vstack([extract_features(img, method='LBP') for img in X_val])
    X_test_lbp = np.vstack([extract_features(img, method='LBP') for img in X_test])

    # Hyperparameter tuning
    print("Tuning GLCM model...")
    best_glcm_model = tune_hyperparameters(X_train_glcm, y_train)
    print("Tuning LBP model...")
    best_lbp_model = tune_hyperparameters(X_train_lbp, y_train)

    # Validation set evaluation
    val_accuracy_glcm = accuracy_score(y_val, best_glcm_model.predict(X_val_glcm))
    val_accuracy_lbp = accuracy_score(y_val, best_lbp_model.predict(X_val_lbp))
    print(f"Optimized GLCM validation accuracy: {val_accuracy_glcm}")
    print(f"Optimized LBP validation accuracy: {val_accuracy_lbp}")

    # Test set evaluation
    y_test_pred_glcm = best_glcm_model.predict(X_test_glcm)
    y_test_pred_lbp = best_lbp_model.predict(X_test_lbp)

    test_accuracy_glcm = accuracy_score(y_test, y_test_pred_glcm)
    test_accuracy_lbp = accuracy_score(y_test, y_test_pred_lbp)
    print(f"Optimized GLCM test accuracy: {test_accuracy_glcm}")
    print(f"Optimized LBP test accuracy: {test_accuracy_lbp}")

    # Print classification reports and confusion matrices for GLCM and LBP models
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

    return best_glcm_model, best_lbp_model

# Prediction function
def predict(image, algorithm):
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image.resize((128, 128)))  # Resize image

    # Extract features
    features = extract_features(image, method=algorithm)
    features = np.array(features).reshape(1, -1)  # Convert to 2D array

    # Use selected algorithm for prediction
    model = best_glcm_model if algorithm == 'GLCM' else best_lbp_model
    prediction = model.predict(features)
    return "Grass" if prediction[0] == 0 else "Wood"

# Evaluate entire folder of images
def evaluate_folder(folder_path, algorithm):
    images, labels = load_images_from_folder(folder_path)
    correct_predictions = 0
    total_images = len(images)

    for img, label in zip(images, labels):
        pred = predict(Image.fromarray(img), algorithm)
        correct_predictions += (pred == "Grass" if label == 0 else "Wood")

    accuracy = correct_predictions / total_images
    return f"{algorithm} model accuracy on folder: {accuracy:.2f}"

# Run the main function to train and evaluate models
best_glcm_model, best_lbp_model = main()

# Gradio interface (single image and folder batch evaluation)
iface = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type='pil', label="Upload Image"),
            gr.Dropdown(choices=["GLCM", "LBP"], label="Choose Algorithm")],
    outputs="text",
    live=True,
    title="Texture Classification",
    description="Upload an image and select an algorithm (GLCM or LBP) to classify the texture as Grass or Wood."
)

# Folder evaluation interface
folder_iface = gr.Interface(
    fn=evaluate_folder,
    inputs=[gr.Textbox(label="Input folder path"),
            gr.Dropdown(choices=["GLCM", "LBP"], label="Choose Algorithm")],
    outputs="text",
    title="Batch evaluate images in a folder"
)

# Launch Gradio interfaces
iface.launch()
folder_iface.launch()