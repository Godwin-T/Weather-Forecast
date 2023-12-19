import shutil
from pathlib import Path

DATASET_TYPES = ["test", "train"]
DROP_COLNAMES = ["Date"]
TARGET_COLUMN = "raintomorrow"
RAW_DATASET = "../raw_data/weatherAUS.csv"
PROCESSED_DATASET = "../processed_data/weatherAUS.csv"
PARAMETERS = "../parameters.json"


def delete_and_recreate_dir(path):
    try:
        shutil.rmtree(path)
    except:
        pass
    finally:
        Path(path).mkdir(parents=True, exist_ok=True)

