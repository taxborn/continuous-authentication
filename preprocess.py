"""
Preprocessing code for mauth dataset.

TODO: If I implement this on more datasets, see if I can generalize, but I think specializing this
per dataset is the way to go
"""
import copy
import time
import glob
import config
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from sklearn.model_selection import train_test_split
from collections import deque, defaultdict

def binary_classify(subject: int, save: bool = False) -> list:
    # TODO: Check if FEATURE_FILE exists?
    df = pd.read_csv(config.FEATURE_FILE)

    genuine_user_data = df.loc[df["ID"].isin([subject])]
    genuine_user_data = copy.deepcopy(genuine_user_data.values)
    genuine_user_data[:, 0] = 1

    imposter_user_data = df[df["ID"] != subject].sample(genuine_user_data.shape[0])
    imposter_user_data = copy.deepcopy(imposter_user_data.values)
    imposter_user_data[:, 0] = 0

    dataset = pd.concat([pd.DataFrame(genuine_user_data, columns=df.columns), pd.DataFrame(imposter_user_data, columns=df.columns)])
    dataset.replace([-np.inf, np.inf], 0, inplace=True)
    dataset["ID"] = dataset["ID"].astype('int')
    if save: dataset.to_csv(f"data/proc/binuser_{subject}_data.csv", index=False)

    X = dataset.iloc[:, 1:] # features
    y = dataset.iloc[:, 0] # id

    # TODO: Configurable train/test size?
    return train_test_split(X, y, train_size=0.7)

def preprocess_raw_subject(subject: int) -> pd.DataFrame:
    # TODO: Check if file exists?
    df = pd.read_csv(f"{config.RAW_DATA_FOLDER}/user_{subject}_data.csv")
    # Since this is used a lot, compute once
    dt = df.Timestamp - df.Timestamp.shift(1)

    df = df.assign(X_Speed=lambda row: (row.X - row.X.shift(1)) / dt)
    df = df.assign(Y_Speed=lambda row: (row.Y - row.Y.shift(1)) / dt)
    df = df.assign(Speed=lambda row: np.sqrt((row.X_Speed ** 2) + (row.Y_Speed ** 2)))
    df = df.assign(X_Acceleration=lambda row: (row.X_Speed - row.X_Speed.shift(1)) / dt)
    df = df.assign(Y_Acceleration=lambda row: (row.Y_Speed - row.Y_Speed.shift(1)) / dt)
    df = df.assign(Acceleration=lambda row: (row.Speed - row.Speed.shift(1)) / dt)
    df = df.assign(Jerk=lambda row: (row.Acceleration - row.Acceleration.shift(1)) / dt)

    return df.fillna(0)

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
    data = defaultdict(list)
    window = deque(maxlen=config.SEQUENCE_LENGTH)
    subject = df.iloc[1]['ID']

    for i, row in enumerate(df.values):
        window.append(row)
        if len(window) != config.SEQUENCE_LENGTH: continue
        # TODO: Better print message
        if i == 0: print(f"{row}")
        cpy = np.copy(window)

        data['Mean_X_Speed'].append(cpy[:, 6].mean())
        data['Std_X_Speed'].append(cpy[:, 6].std())
        data['Min_X_Speed'].append(cpy[:, 6].min())
        data['Max_X_Speed'].append(cpy[:, 6].max())

        data['Mean_Y_Speed'].append(cpy[:, 6].mean())
        data['Std_Y_Speed'].append(cpy[:, 6].std())
        data['Min_Y_Speed'].append(cpy[:, 6].min())
        data['Max_Y_Speed'].append(cpy[:, 6].max())

    out = pd.DataFrame.from_dict(data)
    # TODO: Figure out a different way to do this, similar to how we compute features
    out.insert(0, "ID", subject)
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

    TODO: Is tuple() the only way to evaluate?
    """
    subjects = 15
    print(f"Processing the feature set for {subjects} subjects.")
    start = time.time()
    with multiprocessing.Pool(processes=16) as pool:
        tuple(tqdm(pool.imap_unordered(process_subject, range(subjects)), total=subjects))
    print(f"Took {time.time() - start:3.3f}s")

def create_feature_file():
    # TODO: What to do if the file exists? overwrite? ask? increment a value in the name?
    subject_data = glob.glob(f"data/proc/user_*_data.csv")

    print("Copying individual user feature files to master data file...", end=" ")
    start = time.time()
    with open(config.FEATURE_FILE, 'wb') as outfile:
        for i, subject in enumerate(subject_data):
            with open(subject, 'rb') as infile:
                if i != 0:
                    infile.readline()
                if i > 1:
                    outfile.write(b'\n')
                shutil.copyfileobj(infile, outfile)
    print(f"took {time.time() - start:.3f}s")


if __name__ == "__main__":
    multiprocess_all_subjects()
    create_feature_file()