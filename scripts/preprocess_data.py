from typing import List

import os
import pandas as pd
from sklearn.impute import SimpleImputer

from utils_and_constants import (
    DROP_COLNAMES,
    PROCESSED_DATASET,
    RAW_DATASET,
    TARGET_COLUMN,
)


def read_dataset(
    filename: str, drop_columns: List[str], target_column: str
) -> pd.DataFrame:
    """
    Reads the raw data file and returns pandas dataframe
    Target column values are expected in binary format with Yes/No values

    Parameters:
    filename (str): raw data filename
    drop_columns (List[str]): column names that will be dropped
    target_column (str): name of target column

    Returns:
    pd.Dataframe: Target encoded dataframe
    """
    df = pd.read_csv(filename).drop(columns=drop_columns)
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    categorical_cols = df.dtypes[df.dtypes == 'object'].index.tolist()

    for col in categorical_cols:
        df[col] = df[col].str.lower().str.replace(' ', '_')
    df[target_column] = df[target_column].map({"yes": 1, "no": 0})

    return df


# def target_encode_categorical_features(
#     df: pd.DataFrame, categorical_columns: List[str], target_column: str
# ) -> pd.DataFrame:
#     """
#     Target encodes the categorical features of the dataframe

#     Parameters:
#     df (pd.Dataframe): Pandas dataframe containing features and targets
#     categorical_columns (List[str]): categorical column names that will be target encoded
#     target_column (str): name of target column

#     Returns:
#     pd.Dataframe: Target encoded dataframe
#     """
#     encoded_data = df.copy()

#     # Iterate through categorical columns
#     for col in categorical_columns:
#         # Calculate mean target value for each category
#         encoding_map = df.groupby(col)[target_column].mean().to_dict()

#         # Apply target encoding
#         encoded_data[col] = encoded_data[col].map(encoding_map)

#     return encoded_data



def impute_data(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes numerical data to its mean value
    
    Parameters:
    filename (str): raw data filename
    drop_columns (List[str]): column names that will be dropped
    target_column (str): name of target column

    Returns:
    pd.Dataframe: Imputed and Scaled dataframe
    """

    # Impute data with mean strategy
    numerical_imputer = SimpleImputer(strategy="mean")
    categorical_imputer = SimpleImputer(strategy="most_frequent")

    numerical_col = df_features.select_dtypes(exclude=['object']).columns.tolist()
    numerical_col.remove(TARGET_COLUMN)
    categorical_col = df_features.select_dtypes(include=['object']).columns.tolist()
    categorical_col.append(TARGET_COLUMN)
    
    df_features[numerical_col] = numerical_imputer.fit_transform(df_features[numerical_col].values)
    df_features[categorical_col] = categorical_imputer.fit_transform(df_features[categorical_col].values)

    return df_features


def main():
    # Read data
    weather = read_dataset(
        filename=RAW_DATASET, drop_columns=DROP_COLNAMES, target_column=TARGET_COLUMN
    )
  
    # Impute & Write processed dataset
    weather_features_processed = impute_data(weather)

    if not os.path.dirname(PROCESSED_DATASET):
        os.mkdir(os.path.dirname())
        
    weather_features_processed.to_csv(PROCESSED_DATASET, index=None)

if __name__ == "__main__":
    main()
