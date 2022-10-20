import pandas as pd
import numpy as np
from src.config import NUMBER_OF_TEST


def LotFrontage_points(lotfrontage_) -> int:
    if lotfrontage_ >= 100:
        return 5
    elif 40 <= lotfrontage_ < 100:
        return 4
    elif 30 <= lotfrontage_ < 40:
        return 3
    elif 0 < lotfrontage_ < 30:
        return 2
    else:
        return 0


def LotArea_points(lotarea_) -> int:
    if lotarea_ >= 16000:
        return 5
    elif 9000 <= lotarea_ < 16000:
        return 4
    elif 5000 <= lotarea_ < 9000:
        return 3
    elif 0 < lotarea_ < 5000:
        return 2
    else:
        return 0


def Utilities_points(utilities_) -> int:
    if utilities_ == 'AllPub':
        return 5
    elif utilities_ == 'NoSewr':
        return 4
    elif utilities_ == 'NoSeWa':
        return 3
    elif utilities_ == 'ELO':
        return 2
    else:
        return 0


def OverallQual_points(overallqual_) -> int:
    excellent = [10, 9, 8]
    good = [7, 6]
    average = [5, 4]
    poor = [3, 2, 1]
    if overallqual_ in excellent:
        return 5
    elif overallqual_ in good:
        return 4
    elif overallqual_ in average:
        return 3
    elif overallqual_ in poor:
        return 2
    else:
        return 0


def OverallCond_points(overallcond_) -> int:
    excellent = [10, 9, 8]
    good = [7, 6]
    average = [5, 4]
    poor = [3, 2, 1]
    if overallcond_ in excellent:
        return 5
    elif overallcond_ in good:
        return 4
    elif overallcond_ in average:
        return 3
    elif overallcond_ in poor:
        return 2
    else:
        return 0


def YearBuilt_points(yearbuilt_) -> int:
    if yearbuilt_ >= 2000:
        return 5
    elif 1971 <= yearbuilt_ < 2000:
        return 4
    elif 1920 <= yearbuilt_ < 1971:
        return 3
    elif yearbuilt_ < 1920:
        return 2
    else:
        return 0


def BsmtFinType1_points(bsmtfintype1_) -> int:
    if bsmtfintype1_ == 'GLQ' or bsmtfintype1_ == 'ALQ':
        return 5
    elif bsmtfintype1_ == 'BLQ' or bsmtfintype1_ == 'Rec':
        return 4
    elif bsmtfintype1_ == 'LwQ':
        return 3
    elif bsmtfintype1_ == 'Unf':
        return 2
    else:
        return 0


def BsmtFinType2_points(bsmtfintype2_) -> int:
    if bsmtfintype2_ == 'GLQ' or bsmtfintype2_ == 'ALQ':
        return 5
    elif bsmtfintype2_ == 'BLQ' or bsmtfintype2_ == 'Rec':
        return 4
    elif bsmtfintype2_ == 'LwQ':
        return 3
    elif bsmtfintype2_ == 'Unf':
        return 2
    else:
        return 0


def TotalBsmtSF_points(totalbsmtsf_) -> int:
    if totalbsmtsf_ >= 3000:
        return 5
    elif 1000 <= totalbsmtsf_ < 3000:
        return 4
    elif 500 <= totalbsmtsf_ < 1000:
        return 3
    elif 0 < totalbsmtsf_ < 500:
        return 2
    else:
        return 0


def TotRmsAbvGrd_points(totrmsabvgrd_) -> int:
    if totrmsabvgrd_ >= 10:
        return 5
    elif 6 <= totrmsabvgrd_ < 10:
        return 4
    elif 3 <= totrmsabvgrd_ < 6:
        return 3
    elif totrmsabvgrd_ < 3:
        return 2
    else:
        return 0


def GarageType_points(garagetype_) -> int:
    if garagetype_ == '2Types' or garagetype_ == 'Attchd':
        return 5
    elif garagetype_ == 'Basment' or garagetype_ == 'BuiltIn':
        return 4
    elif garagetype_ == 'CarPort':
        return 3
    elif garagetype_ == 'Detchd':
        return 2
    else:
        return 0


def GarageYrBlt_points(garageyrblt_) -> int:
    if garageyrblt_ >= 2000:
        return 5
    elif 1971 <= garageyrblt_ < 2000:
        return 4
    elif 1920 <= garageyrblt_ < 1971:
        return 3
    elif garageyrblt_ < 1920:
        return 2
    else:
        return 0


def GarageArea_points(garagearea_) -> int:
    if garagearea_ >= 900:
        return 5
    elif 400 <= garagearea_ < 900:
        return 4
    elif 200 <= garagearea_ < 400:
        return 3
    elif 0 < garagearea_ < 200:
        return 2
    else:
        return 0


def PoolArea_points(poolarea_) -> int:
    if poolarea_ >= 100:
        return 5
    elif 50 <= poolarea_ < 100:
        return 4
    elif 2 <= poolarea_ < 50:
        return 3
    elif 0 < poolarea_ < 2:
        return 2
    else:
        return 0


def MiscFeature_points(miscfeature_) -> int:
    if miscfeature_ != 'NA':
        return 5
    else:
        return 0


def class_of_building(df: pd.DataFrame) -> pd.DataFrame:
    building_class = []
    for index, row in df.iterrows():
        count = 0
        count += LotFrontage_points(row['LotFrontage'])
        count += LotArea_points(row['LotArea'])
        count += Utilities_points(row['Utilities'])
        count += OverallQual_points(row['OverallQual'])
        count += OverallCond_points(row['OverallCond'])
        count += YearBuilt_points(row['YearBuilt'])
        count += BsmtFinType1_points(row['BsmtFinType1'])
        count += BsmtFinType2_points(row['BsmtFinType2'])
        count += TotalBsmtSF_points(row['TotalBsmtSF'])
        count += TotRmsAbvGrd_points(row['TotRmsAbvGrd'])
        count += GarageType_points(row['GarageType'])
        count += GarageYrBlt_points(row['GarageYrBlt'])
        count += GarageArea_points(row['GarageArea'])
        count += PoolArea_points(row['PoolArea'])
        count += MiscFeature_points(row['MiscFeature'])
        building_class.append(round(count / NUMBER_OF_TEST))
    i = 0
    for index, row in df.iterrows():
        df.loc[index, 'ClassBuilt'] = building_class[i]
        i += 1
    df['ClassBuilt'] = df['ClassBuilt'].astype(np.int32)
    return df


def age_of_building_at_the_time_of_sale(df: pd.DataFrame) -> pd.DataFrame:
    building_age = []
    for index, row in df.iterrows():
        building_age.append(int(row['YrSold'] - row['YearBuilt']))
    i = 0
    for index, row in df.iterrows():
            df.loc[index,'BuiltAge'] = building_age[i]
            i += 1
    df['BuiltAge'] = df['BuiltAge'].astype(np.int32)
    return df


def feature_generation(df: pd.DataFrame) -> pd.DataFrame:
    df = class_of_building(df)
    df = age_of_building_at_the_time_of_sale(df)
    return df
