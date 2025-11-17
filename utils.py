from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def print_classification_metrics(y_true, y_pred):
    """
    Computes and prints standard classification metrics.

    Metrics include:
    - Accuracy
    - Weighted Precision
    - Weighted Recall
    - Weighted F1 Score

    Args:
        y_true (array-like): Ground truth class labels.
        y_pred (array-like): Predicted class labels from the model.

    Returns:
        None. Prints metrics to stdout.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print("=== Evaluation Metrics ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
