"""
Preprocessing code for mauth dataset.

TODO: If I implement this on more datasets, see if I can generalize, but I think specializing this
per dataset is the way to go
"""
import copy
import glob
import config
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from sklearn.model_selection import train_test_split  # type: ignore
from collections import deque, defaultdict


def binary_classify(subject: int, save: bool = False, shuffle: bool = True) -> list:
    # TODO: Check if FEATURE_FILE exists?
    df = pd.read_csv(config.FEATURE_FILE)

    genuine_user_data = df.loc[df["ID"].isin([subject])]
    genuine_data = copy.deepcopy(genuine_user_data.values)
    genuine_data[:, 0] = 1

    imposter_user_data = df[df["ID"] != subject].sample(
        genuine_data.shape[0], random_state=config.RANDOM_STATE
    )
    imposter_data = copy.deepcopy(imposter_user_data.values)
    imposter_data[:, 0] = 0

    dataset = pd.concat(
        [
            pd.DataFrame(genuine_data, columns=df.columns),
            pd.DataFrame(imposter_data, columns=df.columns),
        ]
    )
    dataset.replace([-np.inf, np.inf], 0, inplace=True)
    dataset["ID"] = dataset["ID"].astype("int")
    if save:
        dataset.to_csv(f"data/proc/binuser_{subject}_data.csv", index=False)

    X = dataset.iloc[:, 1:]  # features
    y = dataset.iloc[:, 0]  # id

    # TODO: Configurable train/test size?
    return train_test_split(
        X, y, train_size=0.9, shuffle=shuffle, random_state=config.RANDOM_STATE
    )


def preprocess_raw_subject(subject: int) -> pd.DataFrame:
    # TODO: Check if file exists?
    df = pd.read_csv(f"{config.RAW_DATA_FOLDER}/user_{subject}_data.csv")
    # Since this is used a lot, compute once
    dt = df["Timestamp"].diff()

    df["X_Speed"] = df["X"].diff() / dt
    df["Y_Speed"] = df["Y"].diff() / dt
    df["Speed"] = np.sqrt(df["X_Speed"] ** 2 + df["Y_Speed"] ** 2)
    df["X_Acceleration"] = df["X_Speed"].diff() / dt
    df["Y_Acceleration"] = df["Y_Speed"].diff() / dt
    df["Acceleration"] = df["Speed"].diff() / dt
    df["Jerk"] = df["Acceleration"].diff() / dt

    # Since really the only NaNs we see are the the first few rows of speed, acceleration, and jerk, due to them not having
    # info, it seems reasonable to start this off at 0
    return df.fillna(0)


def calculate_statistics(
    data: dict[str, list], array: np.ndarray, stats: dict[str, int]
):
    """
    Calculate mean, std, min, and max for the specified feature and append to the data dictionary.
    """

    for stat, column_index in stats.items():
        slice = array[:, column_index]

        data[f"Mean_{stat}"].append(slice.mean())
        data[f"Std_{stat}"].append(slice.std())
        data[f"Min_{stat}"].append(slice.min())
        data[f"Max_{stat}"].append(slice.max())


def process_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the features for a given DataFrame. We expect the dataframe to come from preprocess_raw_subject() to get some basic features set up
    for use later.

    Indexes:
    0: ID
    1: Timestamp
    2: X
    3: Y
    4: Button
    5: Duration
    6-12: Set in preprocess_raw_subject()
    """
    data_dictionary: dict[str, list] = defaultdict(list)
    rolling_window: deque[list] = deque(maxlen=config.SEQUENCE_LENGTH)

    stats = {
        "X_Speed": 6,
        "Y_Speed": 7,
        "Speed": 8,
        "X_Acceleration": 9,
        "Y_Acceleration": 10,
        "Acceleration": 11,
        "Jerk": 12,
    }

    for raw_row in df.values:
        rolling_window.append(raw_row)
        if len(rolling_window) != config.SEQUENCE_LENGTH:
            continue
        row = np.copy(rolling_window)

        # Some statistics surrounding velocity, acceleration, and jerk
        calculate_statistics(data_dictionary, row, stats)

    out = pd.DataFrame.from_dict(data_dictionary)
    # Re-insert the subject's ID back into the dataframe
    out.insert(0, "ID", df.iloc[1]["ID"].astype("int"))
    return out


def process_subject(subject: int):
    """
    Process a specific subject. This will also save the resulting DataFrame to a CSV
    """
    df = preprocess_raw_subject(subject)
    df = process_features(df)
    # TODO: What to do if the file exists? overwrite? ask? increment a value in the name?
    df.to_csv(f"data/proc/user_{subject}_data.csv", index=False)


def multiprocess_all_subjects():
    """
    Processes the complete feature set for all subjects, utilizing multiprocessing to make this go *a bit* faster.

    https://stackoverflow.com/questions/68065937/how-to-show-progress-bar-tqdm-while-using-multiprocessing-in-python

    To have a pretty iterator with multiprocessing, we need to lazily evaluate the function we want to map over.
    To actually execute the operations out of their lazy state, we wrap everything in tuple().

    TODO: Is tuple() the best way to evaluate?
    """
    subjects = 15
    with multiprocessing.Pool(processes=subjects) as pool:
        tuple(
            tqdm(
                pool.imap_unordered(process_subject, range(subjects)),
                total=subjects,
                unit="subject",
                desc="Processing the feature sets",
            )
        )


def create_feature_file():
    # TODO: What to do if the file exists? overwrite? ask? increment a value in the name?
    subject_data = glob.glob("data/proc/user_*_data.csv")

    with open(config.FEATURE_FILE, "wb") as outfile:
        for i, subject in enumerate(subject_data):
            with open(subject, "rb") as infile:
                if i != 0:
                    infile.readline()
                if i > 1:
                    outfile.write(b"\n")
                shutil.copyfileobj(infile, outfile)
