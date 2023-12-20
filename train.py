import config
import pathlib
import preprocess
import validation
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # type: ignore

if __name__ == "__main__":
    if not pathlib.Path(config.FEATURE_FILE).exists():
        print(f"{config.FEATURE_FILE} does not exist, computing features.")
        preprocess.multiprocess_all_subjects()
        preprocess.create_feature_file()

    X_train, X_test, y_train, y_test = preprocess.binary_classify(0)

    classifier = KNeighborsClassifier(metric='cityblock', n_jobs=-1, n_neighbors=3)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print("KNN Classifier validation: ")
    validation.calculate_validations(y_test, y_pred)

    regressor = KNeighborsRegressor(metric='cityblock', n_jobs=-1, n_neighbors=3)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    # We start with a trust score of 100
    C = 100
    violations = 0

    for pred in tqdm(y_pred, unit=" Predictions", desc="Calculating trust score"):
        violations += C < config.THRESHOLD

        if pred >= 0.5: C = min(C + pred, 100)
        elif pred >= 0.3: C = max(C - (1 - pred), 0)
        else: C = max(C - 1, 0)

    print(f"\nKNN Regressor results (trust score): {C = :3.2f} {violations = }")