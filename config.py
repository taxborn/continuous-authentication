# The width of the sliding window throughout the raw data to compute features over
SEQUENCE_LENGTH = 384
RAW_DATA_FOLDER = "data/raw"
# TODO: Encode something about the feature set in the name?
FEATURE_FILE = f"data/proc/features_SQ{SEQUENCE_LENGTH}.csv"
# Set to None to disable, or set to a number to make the preprocessing
# and models deterministic
RANDOM_STATE = 4

# Trust score (scale from 0-100)
STARTING_VALUE = 100
THRESHOLD = 70
