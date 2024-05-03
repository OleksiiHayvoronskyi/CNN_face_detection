# REQUIRED LIBRARIES

# For numerical operations
import numpy as np
# For computing classification metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score
# For computing additional classification metrics
from sklearn.metrics import f1_score, roc_auc_score
# For data visualization
import matplotlib.pyplot as plt
# For evaluating classification performance
from sklearn.metrics import confusion_matrix, classification_report


# Making predictions on test_data
def compute_metrics_and_visualizations(model, X_test, y_test, history):
    # Making predictions on test_data
    predictions = model.predict(X_test)

    # Convert one-hot encoded labels back to class labels
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels,
                                average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    roc_auc = roc_auc_score(y_test, predictions, average='weighted',
                            multi_class='ovr')

    # Print metrics
    print(f"\nAccuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"ROC AUC Score: {roc_auc:.3f}\n")

    # Plot metrics
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    metrics_values = [accuracy, precision, recall, f1, roc_auc]

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.bar(metrics_names, metrics_values,
                  color=['blue', 'green', 'orange', 'red', 'purple'])
    ax.set_xlabel('Metrics', color='navy', size=12)
    ax.set_ylabel('Values', color='navy', size=12)
    ax.set_title('Performance Metrics after training', color='navy', size=14)
    ax.set_ylim([0, 1])

    for bar, value in zip(bars, metrics_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f'{value:.3f}',
                ha='center', va='center', color='white', fontweight='bold')

    plt.show()

    # Plot loss if history is not None
    if history is not None:
        plt.plot(history.history['loss'], 'r', label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.title('Training and Validation Loss', color='navy', size=14)
        plt.xlabel('Epochs', color='navy', size=12)
        plt.ylabel('Loss', color='navy', size=12)
        plt.legend()
        plt.show()

        # Plot accuracy
        plt.plot(history.history['accuracy'], 'r', label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy', color='navy', size=14)
        plt.xlabel('Epoch', color='navy', size=12)
        plt.ylabel('Accuracy', color='navy', size=12)
        plt.legend()
        plt.show()

        # Accuracy of the model.
        print(model.evaluate(X_test, y_test))

        # Generate confusion matrix
        y_pred = np.argmax(model.predict(X_test), axis=1)
        # cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
        # plt.imshow(cm, cmap=plt.cm.Blues)
        # plt.title('Confusion Matrix', color='navy', size=14)
        # plt.colorbar()
        # plt.xlabel('Predicted labels', color='navy', size=12)
        # plt.ylabel('True labels', color='navy', size=12)
        # plt.show()

        # Generate classification report
        print('Classification Report:')
        print(classification_report(np.argmax(y_test, axis=1), y_pred))
