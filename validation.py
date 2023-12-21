from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # type: ignore


def calculate_validations(y_test, y_pred):
    print(f"Accuracy: {100 * accuracy_score(y_test, y_pred):3.4f}%")
    print(f"Precision: {100 * precision_score(y_test, y_pred):3.4f}%")
    print(f"Recall: {100 * recall_score(y_test, y_pred):3.4f}%")
    print(f"F1: {100 * f1_score(y_test, y_pred):3.4f}%")
