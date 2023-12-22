from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score  # type: ignore
import config
from tqdm import tqdm

def calculate_metric_scores(y_test, y_pred):
    # TODO: I probably want to return these values instead of printing
    print(f"Accuracy: {100 * accuracy_score(y_test, y_pred):3.4f}%")
    print(f"Precision: {100 * precision_score(y_test, y_pred):3.4f}%")
    print(f"Recall: {100 * recall_score(y_test, y_pred):3.4f}%")
    print(f"F1: {100 * f1_score(y_test, y_pred):3.4f}%")

def calculate_trust_score(y_pred: list[int]) -> tuple[int, int]:
    C = config.STARTING_VALUE
    # TODO: This isn't needed, once a violation occurs we re-authenticate and 
    violations = 0

    for pred in tqdm(y_pred, unit=" Predictions", desc="Calculating trust score"):
        violations += C < config.THRESHOLD

        if pred >= 0.5:
            C = min(C + pred, 100)
        elif pred >= 0.3:
            C = max(C - (1 - pred), 0)
        else:
            C = max(C - 1, 0)

    return C, violations