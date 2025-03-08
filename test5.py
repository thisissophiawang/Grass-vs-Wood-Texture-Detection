import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.transform import rotate, rescale
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import os
import gradio as gr

# Enhanced GLCM feature extraction
def extract_glcm_features(image, distances=[1, 2, 3, 4], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    features = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in properties:
        features.append(graycoprops(glcm, prop).flatten())
    return np.hstack(features)

# Enhanced LBP feature extraction
def extract_lbp_features(image, radius=3, n_points=24):
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2), density=True)
    return hist

# Combined feature extraction
def extract_features(image, glcm_distances=[1, 2, 3, 4], glcm_angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], lbp_radius=3, lbp_n_points=24):
    glcm_features = extract_glcm_features(image, distances=glcm_distances, angles=glcm_angles)
    lbp_features = extract_lbp_features(image, radius=lbp_radius, n_points=lbp_n_points)
    return np.hstack([glcm_features, lbp_features])

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
        img = img.resize((128, 128))
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

# Visualization functions
def plot_feature_distribution(X, y, feature_names):
    plt.figure(figsize=(20, 15))
    for i, feature in enumerate(feature_names):
        plt.subplot(4, 4, i+1)
        plt.hist(X[y==0, i], alpha=0.5, label='Grass')
        plt.hist(X[y==1, i], alpha=0.5, label='Wood')
        plt.title(feature)
        plt.legend()
    plt.tight_layout()
    plt.show()

def plot_decision_boundary(X, y, model, model_name):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_pca[y==0, 0], X_pca[y==0, 1], c='blue', label='Grass')
    plt.scatter(X_pca[y==1, 0], X_pca[y==1, 1], c='red', label='Wood')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(f'Decision Boundary - {model_name}')
    plt.legend()
    plt.show()

# Main function
def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset('lab2 image')

    # Extract features
    print("Extracting features...")
    X_train_features = np.array([extract_features(img) for img in X_train])
    X_val_features = np.array([extract_features(img) for img in X_val])
    X_test_features = np.array([extract_features(img) for img in X_test])

    # Visualize feature distribution
    feature_names = ['GLCM Contrast', 'GLCM Dissimilarity', 'GLCM Homogeneity', 
                     'GLCM Energy', 'GLCM Correlation', 'GLCM ASM', 'LBP Histogram']
    plot_feature_distribution(X_train_features, y_train, feature_names)

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
    svm_grid = GridSearchCV(svm, svm_params, cv=5, n_jobs=-1)
    svm_grid.fit(X_train_resampled, y_train_resampled)
    svm_model = svm_grid.best_estimator_
    
    print("Tuning Random Forest model...")
    rf_grid = GridSearchCV(rf, rf_params, cv=5, n_jobs=-1)
    rf_grid.fit(X_train_resampled, y_train_resampled)
    rf_model = rf_grid.best_estimator_

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

        # Plot decision boundary
        plot_decision_boundary(X_train_scaled, y_train, model, name)

    return ensemble_model, scaler

# Prediction function for Gradio interface
def predict_with_params(image, model_choice, glcm_distance, glcm_angle, lbp_radius, lbp_points):
    image = image.convert('L')
    image = np.array(image.resize((128, 128)))
    
    features = extract_features(image, 
                                glcm_distances=[glcm_distance], 
                                glcm_angles=[glcm_angle], 
                                lbp_radius=lbp_radius, 
                                lbp_n_points=lbp_points)
    
    features_scaled = feature_scaler.transform(features.reshape(1, -1))
    
    if model_choice == "SVM":
        prediction = svm_model.predict(features_scaled)
    elif model_choice == "Random Forest":
        prediction = rf_model.predict(features_scaled)
    else:
        prediction = ensemble_model.predict(features_scaled)
    
    return "Grass" if prediction[0] == 0 else "Wood"

# Run main function to train and evaluate models
ensemble_model, feature_scaler = main()

# Gradio interface
iface = gr.Interface(
    fn=predict_with_params,
    inputs=[
        gr.Image(type='pil'),
        gr.Radio(["SVM", "Random Forest", "Ensemble"], label="Choose Model"),
        gr.Slider(1, 5, value=1, step=1, label="GLCM Distance"),
        gr.Slider(0, np.pi, value=0, step=np.pi/4, label="GLCM Angle"),
        gr.Slider(1, 5, value=3, step=1, label="LBP Radius"),
        gr.Slider(8, 24, value=24, step=8, label="LBP Points")
    ],
    outputs="text",
    live=True,
    title="Advanced Texture Classification",
    description="Upload an image and adjust parameters to classify the texture as Grass or Wood."
)

# Launch Gradio interface
iface.launch()