# The width of the sliding window throughout the raw data to compute features over
SEQUENCE_LENGTH = 128
RAW_DATA_FOLDER = "data/raw"
# TODO: Encode something about the feature set in the name?
FEATURE_FILE = f"data/proc/features_SQ{SEQUENCE_LENGTH}.csv"

# Trust score (scale from 0-100)
THRESHOLD = 70