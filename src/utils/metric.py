import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix


def evaluate_model(model, validation_set, category_labels):
    """
    Evaluate a trained Keras model on a validation set and print classification metrics.
    Args:
        model: Trained Keras model
        validation_set: Validation data generator
        category_labels: True labels (one-hot or categorical)
    Returns:
        metrics: dict with accuracy, precision, recall, f1, and classification report
    """
    # Get true labels for validation samples
    true_labels = category_labels[validation_set.index_array]
    true_labels = true_labels.astype(int)

    # Predict probabilities
    prediction = model.predict(validation_set)
    binary_predictions = (prediction > 0.5).astype(int)

    # Compute metrics
    report = classification_report(true_labels, binary_predictions, output_dict=True)
    accuracy = accuracy_score(true_labels, binary_predictions)
    precision = precision_score(true_labels, binary_predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, binary_predictions, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, binary_predictions, average='weighted', zero_division=0)
    mcm = multilabel_confusion_matrix(true_labels, binary_predictions)

    print(classification_report(true_labels, binary_predictions))
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1)
    print('Multilabel Confusion Matrix:', mcm)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': report,
        'multilabel_confusion_matrix': mcm
    }

# Example usage:
# metrics = evaluate_model(MobileNetV2_model, validation_set, category_labels)
# metrics = evaluate_model(CNN_model, validation_set, category_labels)
# metrics = evaluate_model(InceptionV3_model, validation_set, category_labels)
