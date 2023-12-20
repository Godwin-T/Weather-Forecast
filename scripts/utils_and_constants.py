import os
import shutil
from pathlib import Path

if os.getenv('DVC'):
    DATASET_TYPES = ["test", "train"]
    DROP_COLNAMES = ["Date"]
    TARGET_COLUMN = "raintomorrow"
    RAW_DATASET = "./raw_data/weatherAUS.csv"
    PROCESSED_DATASET = "./processed_data/weatherAUS.csv"
    PARAMETERS = "./parameters.json"
    MODEL_PATH = "./models/model.pkl"
    METRICS_PATH = "./metrics.json"
    PREDICTIONS_PATH = "./predictions.csv"
    ROC_CURVE_PATH = "./roc_curve.csv"
else:
    DATASET_TYPES = ["test", "train"]
    DROP_COLNAMES = ["Date"]
    TARGET_COLUMN = "raintomorrow"
    RAW_DATASET = "../raw_data/weatherAUS.csv"
    PROCESSED_DATASET = "../processed_data/weatherAUS.csv"
    MODEL_PATH = "../models/model.pkl"
    PARAMETERS = "../parameters.json"
    METRICS_PATH = "../metrics.json"
    PREDICTIONS_PATH = "../predictions.csv"
    ROC_CURVE_PATH = "../roc_curve.csv"


def delete_and_recreate_dir(path):
    try:
        shutil.rmtree(path)
    except:
        pass
    finally:
        Path(path).mkdir(parents=True, exist_ok=True)