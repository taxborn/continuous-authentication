import time
import config
import pathlib
import preprocess
import validation
from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":
    if not pathlib.Path(config.FEATURE_FILE).exists():
        print(f"{config.FEATURE_FILE} does not exist, computing features.")
        preprocess.multiprocess_all_subjects()
        preprocess.create_feature_file()

    X_train, X_test, y_train, y_test = preprocess.binary_classify(0)

    classifier = KNeighborsClassifier(metric='cityblock', n_jobs=-1, n_neighbors=3)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    validation.calculate_validations(y_test, y_pred)