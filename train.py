import config
import pathlib
import preprocess

if __name__ == "__main__":
    if not pathlib.Path(config.FEATURE_FILE).exists():
        print(f"{config.FEATURE_FILE} does not exist, computing features.")
        preprocess.multiprocess_all_subjects()
        preprocess.create_feature_file()

    X_train, X_test, y_train, y_test = preprocess.binary_classify(0)