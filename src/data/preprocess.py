import pandas as pd
import numpy as np
from src.config import *


def pre_process_target(df: pd.DataFrame) -> pd.DataFrame:
    df[TARGET_COL] = df[TARGET_COL].astype(np.int32)
    return df


def extract_target(df: pd.DataFrame):
    df, target = df.drop(TARGET_COL, axis=1), df[TARGET_COL]
    return df, target


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df[CAT_COLS] = df[CAT_COLS].astype('category')
    df[REAL_COLS] = df[REAL_COLS].astype(np.int32)
    df[DATA_COLS] = df[DATA_COLS].astype(np.int32)
    return df


def set_idx(df: pd.DataFrame, idx_col: str) -> pd.DataFrame:
    df = df.set_index(idx_col)
    return df


def delete_nan_value(df: pd.DataFrame) -> pd.DataFrame:
    for col in BASEMENT_COLS:
        temp_array = list(df[col])
        count = 0
        for index, row in df.iterrows():
            if row['BsmtQual'] == 'NA':
                if pd.isna(row[col]):
                    temp_array[count] = 'NA'
            count += 1
        df = df.drop(columns=col)
        count = 0
        for index, row in df.iterrows():
            df.loc[index, col] = temp_array[count]
            count += 1
        df = df[(pd.isna(df[col])) == False]

    for col in MASONRY_COLS:
        temp_array = list(df[col])
        count = 0
        for index, row in df.iterrows():
            if row['MasVnrType'] == 'None':
                if pd.isna(row[col]):
                    temp_array[count] = 0
            count += 1
        df = df.drop(columns=col)
        count = 0
        for index, row in df.iterrows():
            df.loc[index, col] = temp_array[count]
            count += 1
        df = df[(pd.isna(df[col])) == False]

    df = df[(pd.isna(df['Electrical'])) == False]

    for col in GARAGE_COLS:
        temp_array = list(df[col])
        count = 0
        for index, row in df.iterrows():
            if row['GarageType'] == 'NA':
                if pd.isna(row[col]):
                    temp_array[count] = 'NA'
            count += 1
        df = df.drop(columns=col)
        count = 0
        for index, row in df.iterrows():
            df.loc[index, col] = temp_array[count]
            count += 1
        df = df[(pd.isna(df[col])) == False]

    for col in GARAGE_COL:
        temp_array = list(df[col])
        count = 0
        for index, row in df.iterrows():
            if row['GarageType'] == 'NA':
                if pd.isna(row[col]):
                    temp_array[count] = 0
            count += 1
        df = df.drop(columns=col)
        count = 0
        for index, row in df.iterrows():
            df.loc[index, col] = temp_array[count]
            count += 1
        df = df[(pd.isna(df[col])) == False]

    return df


def full_columns_with_max_number_of_nan(df: pd.DataFrame) -> pd.DataFrame:
    df[MAX_NUMBER_NAN_VALUE_COLS] = df[MAX_NUMBER_NAN_VALUE_COLS].replace(np.nan, 'NA')
    df['LotFrontage'] = df['LotFrontage'].replace(np.nan, 0)
    df['MasVnrType'] = df['MasVnrType'].replace(np.nan, 'None')

    return df


def pre_process_df(df: pd.DataFrame) -> pd.DataFrame:
    df = set_idx(df, ID_COL)
    df = full_columns_with_max_number_of_nan(df)
    df = delete_nan_value(df)
    df = cast_types(df)
    return df


def pre_process_val(df: pd.DataFrame) -> pd.DataFrame:
    df = set_idx(df, ID_COL)
    df = full_columns_with_max_number_of_nan(df)
    df = delete_nan_value(df)
    df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))
    df = cast_types(df)
    return df

