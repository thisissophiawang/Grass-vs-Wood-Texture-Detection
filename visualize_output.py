import matplotlib.pyplot as plt
import numpy as np

# SVM, Random Forest, and Ensemble output values from your test results
output_data = {
    "SVM": {
        "accuracy": 0.8316,
        "precision": [1.00, 0.73],
        "recall": [0.70, 1.00],
        "f1-score": [0.82, 0.84],
        "support": [312, 252],
        "confusion_matrix": np.array([[217, 95], [0, 252]])
    },
    "Random Forest": {
        "accuracy": 0.8298,
        "precision": [1.00, 0.72],
        "recall": [0.69, 1.00],
        "f1-score": [0.82, 0.84],
        "support": [312, 252],
        "confusion_matrix": np.array([[216, 96], [0, 252]])
    },
    "Ensemble": {
        "accuracy": 0.8316,
        "precision": [1.00, 0.73],
        "recall": [0.70, 1.00],
        "f1-score": [0.82, 0.84],
        "support": [312, 252],
        "confusion_matrix": np.array([[217, 95], [0, 252]])
    }
}

# Plot Precision, Recall, F1-Score for each model
def plot_precision_recall_f1(output_data):
    models = ["SVM", "Random Forest", "Ensemble"]
    metrics = ["precision", "recall", "f1-score"]
    classes = ["Grass", "Wood"]
    
    for metric in metrics:
        plt.figure(figsize=(8, 5))
        for model in models:
            values = output_data[model][metric]
            plt.bar([f"{model}-{cls}" for cls in classes], values, label=model)
        plt.title(f"{metric.capitalize()} Comparison")
        plt.xlabel("Class")
        plt.ylabel(f"{metric.capitalize()}")
        plt.legend(models)
        plt.tight_layout()
        plt.show()

# Plot accuracy comparison for all models
def plot_accuracy_comparison(output_data):
    models = list(output_data.keys())
    accuracies = [output_data[model]['accuracy'] for model in models]
    
    plt.figure(figsize=(8, 5))
    plt.bar(models, accuracies)
    plt.title("Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()

# Plot confusion matrix for each model
def plot_confusion_matrix(output_data):
    models = ["SVM", "Random Forest", "Ensemble"]
    
    for model in models:
        cm = output_data[model]["confusion_matrix"]
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"{model} Confusion Matrix")
        plt.colorbar()
        classes = ["Grass", "Wood"]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f"{cm[i, j]} ({cm_normalized[i, j]*100:.2f}%)",
                         horizontalalignment="center",
                         color="white" if cm[i, j] > cm.max() / 2. else "black")
        
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.show()

# Run all visualizations
plot_precision_recall_f1(output_data)
plot_accuracy_comparison(output_data)
plot_confusion_matrix(output_data)
