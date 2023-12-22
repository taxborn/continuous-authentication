import config
import pathlib
import preprocess
import validation
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # type: ignore

if __name__ == "__main__":
    if not pathlib.Path(config.FEATURE_FILE).exists():
        print(f"{config.FEATURE_FILE} does not exist, computing features.")
        preprocess.multiprocess_all_subjects()
        preprocess.create_feature_file()

    X_train, X_test, y_train, y_test = preprocess.binary_classify(subject=0)

    classifier = KNeighborsClassifier(metric="cityblock", n_jobs=-1, n_neighbors=3)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print("KNN Classifier validation: ")
    validation.calculate_metric_scores(y_test, y_pred)

    regressor = KNeighborsRegressor(metric="cityblock", n_jobs=-1, n_neighbors=3)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    # We start with a trust score of 100
    C, violations = validation.calculate_trust_score(y_pred)
    print(f"Trust score calculations: {C = } {violations = }")
