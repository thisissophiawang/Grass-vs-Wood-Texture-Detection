import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.transform import rotate, rescale
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os
import gradio as gr

# Enhanced GLCM feature extraction
def extract_glcm_features(image, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    features = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in properties:
        features.append(graycoprops(glcm, prop).flatten())
    return np.hstack(features)

# Enhanced multi-scale LBP feature extraction
def extract_multiscale_lbp_features(image):
    radii = [1, 2, 3]
    n_points = [8, 16, 24]
    features = []
    for radius, points in zip(radii, n_points):
        lbp = local_binary_pattern(image, points, radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, points + 3), range=(0, points + 2))
        features.append(hist)
    return np.hstack(features)

# Histogram of Oriented Gradients (HOG)-like feature
def extract_hog_like_features(image, orientations=9, pixels_per_cell=(8, 8)):
    gx = np.gradient(image, axis=0)
    gy = np.gradient(image, axis=1)
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180
    
    hist = np.zeros(orientations)
    for i in range(0, image.shape[0], pixels_per_cell[0]):
        for j in range(0, image.shape[1], pixels_per_cell[1]):
            cell_magnitude = magnitude[i:i+pixels_per_cell[0], j:j+pixels_per_cell[1]]
            cell_orientation = orientation[i:i+pixels_per_cell[0], j:j+pixels_per_cell[1]]
            for o in range(orientations):
                hist[o] += np.sum(cell_magnitude[(cell_orientation >= o*20) & (cell_orientation < (o+1)*20)])
    
    return hist / np.sum(hist)

# Haralick-like texture features
def extract_haralick_like_features(image):
    glcm = graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    features = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in properties:
        features.append(graycoprops(glcm, prop)[0, 0])
    return np.array(features)

# Data augmentation
def augment_image(image):
    augmented = [image]
    augmented.append(rotate(image, angle=90, preserve_range=True).astype(np.uint8))
    augmented.append(rotate(image, angle=180, preserve_range=True).astype(np.uint8))
    augmented.append(rotate(image, angle=270, preserve_range=True).astype(np.uint8))
    augmented.append(rescale(image, 0.5, preserve_range=True, anti_aliasing=True).astype(np.uint8))
    augmented.append(rescale(image, 2, preserve_range=True, anti_aliasing=True).astype(np.uint8))
    return augmented

# Load and augment images from folder
def load_images_from_folder(folder, label=None):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path).convert('L')
        img = img.resize((128, 128))  # Ensure all images are 128x128
        img_array = np.array(img)
        augmented_images = augment_image(img_array)
        images.extend(augmented_images)
        if label is not None:
            labels.extend([label] * len(augmented_images))
    return images, labels

# Load dataset
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

    print(f"Train set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test

# Enhanced feature extraction
def extract_features(image):
    if image.shape != (128, 128):
        image = np.array(Image.fromarray(image).resize((128, 128)))
    glcm_features = extract_glcm_features(image)
    lbp_features = extract_multiscale_lbp_features(image)
    hog_features = extract_hog_like_features(image)
    haralick_features = extract_haralick_like_features(image)
    return np.hstack([glcm_features, lbp_features, hog_features, haralick_features])

# Hyperparameter tuning
def tune_model(model, param_grid, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Main function
def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset('lab2 image')

    # Extract features
    print("Extracting features...")
    X_train_features = np.array([extract_features(img) for img in X_train])
    X_val_features = np.array([extract_features(img) for img in X_val])
    X_test_features = np.array([extract_features(img) for img in X_test])

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_val_scaled = scaler.transform(X_val_features)
    X_test_scaled = scaler.transform(X_test_features)

    # Apply SMOTE for class balancing
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # Define models and parameters
    svm = SVC(probability=True)
    rf = RandomForestClassifier()

    svm_params = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    }
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Train and tune models
    print("Tuning SVM model...")
    svm_model = tune_model(svm, svm_params, X_train_resampled, y_train_resampled)
    
    print("Tuning Random Forest model...")
    rf_model = tune_model(rf, rf_params, X_train_resampled, y_train_resampled)

    # Create ensemble model
    ensemble_model = VotingClassifier(
        estimators=[('svm', svm_model), ('rf', rf_model)],
        voting='soft'
    )
    ensemble_model.fit(X_train_resampled, y_train_resampled)

    # Evaluate models
    models = {
        'SVM': svm_model,
        'Random Forest': rf_model,
        'Ensemble': ensemble_model
    }

    for name, model in models.items():
        val_accuracy = accuracy_score(y_val, model.predict(X_val_scaled))
        test_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
        print(f"{name} Validation Accuracy: {val_accuracy:.4f}")
        print(f"{name} Test Accuracy: {test_accuracy:.4f}")
        print(f"{name} Classification Report:")
        print(classification_report(y_test, model.predict(X_test_scaled), target_names=["Grass", "Wood"]))
        print(f"{name} Confusion Matrix:")
        print(confusion_matrix(y_test, model.predict(X_test_scaled)))
        print()

    return ensemble_model, scaler

# Prediction function
def predict(image, model, scaler):
    image = image.convert('L')
    image = np.array(image.resize((128, 128)))
    features = extract_features(image)
    features_scaled = scaler.transform(features.reshape(1, -1))
    prediction = model.predict(features_scaled)
    return "Grass" if prediction[0] == 0 else "Wood"

# Run main function to train and evaluate models
best_model, feature_scaler = main()

# Gradio interface
iface = gr.Interface(
    fn=lambda img: predict(img, best_model, feature_scaler),
    inputs=gr.Image(type='pil'),
    outputs="text",
    live=True,
    title="Advanced Texture Classification",
    description="Upload an image to classify the texture as Grass or Wood."
)

# Launch Gradio interface
iface.launch()