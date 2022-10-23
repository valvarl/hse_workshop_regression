import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler

from src import utils
from src.config import *


processed_data_path = 'data/processed/'
target_col = 'SalePrice'


def preprocess(input_train_path, input_test_path, output_dir: str = processed_data_path) -> None:
    '''
    :param input_train_path: path to `train.csv`
    :param input_test_path: path to `test.csv`
    :param output_dir: output directory
    '''
    
    train = pd.read_csv(input_train_path, index_col='Id')
    test = pd.read_csv(input_test_path, index_col='Id')
    
    train_target = train[target_col]
    train_data = train.drop(columns=target_col)
    
    merged = pd.concat([train_data, test], axis=0, sort=True)
    
    num_merged = merged.select_dtypes(include = ['int64', 'float64'])
    
    cat_col = ['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold']
    merged.loc[:, cat_col] = merged.loc[:, cat_col].astype('object')
    
    missing_columns = merged.columns[merged.isnull().any()].values
    missing_columns = len(merged) - merged.loc[:, np.sum(merged.isnull())>0].count()

    to_impute_by_none = merged.loc[:, NAN_HAS_SENSE]
    for i in to_impute_by_none.columns:
        merged[i].fillna('None', inplace = True)
        
    # input by mode
    to_impute_by_mode =  merged.loc[:, CAT_COL]
    for i in to_impute_by_mode.columns:
        merged[i].fillna(merged[i].mode()[0], inplace = True)
        
    # input by median
    to_impute_by_median = merged.loc[:, NUM_COL]
    for i in to_impute_by_median.columns:
        merged[i].fillna(merged[i].median(), inplace = True)

    # ohe
    le = LabelEncoder()

    df = merged.reset_index().drop(columns=['Id','LotFrontage'], axis=1)
    df = df.apply(le.fit_transform) # data is converted.

    df['LotFrontage'] = merged['LotFrontage']
    df = df.set_index('LotFrontage').reset_index()
    
    # Impute LotFrontage with median of respective columns (i.e., BldgType)
    merged['LotFrontage'] = merged.groupby(['BldgType'])['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    print('Missing variables left untreated: ', merged.columns[merged.isna().any()].values)  
    
    train_target = np.log1p(train_target)
    
    merged_num = merged.select_dtypes(include = ['int64', 'float64'])
    merged_skewed = np.log1p(merged_num[merged_num.skew()[merged_num.skew() > 0.5].index])


    #Normal variables
    merged_normal = merged_num[merged_num.skew()[merged_num.skew() < 0.5].index]

    #Merging
    merged_num_all = pd.concat([merged_skewed, merged_normal], axis = 1)

    merged_num.update(merged_num_all)

    scaler = RobustScaler()
    merged_num_scaled = scaler.fit_transform(merged_num)
    merged_num_scaled = pd.DataFrame(data = merged_num_scaled, columns = merged_num.columns, index = merged_num.index)
    
    """Let's extract categorical variables first and convert them into category."""
    merged_cat = merged.select_dtypes(include = ['object']).astype('category')

    """let's begin the tedious process of label encoding of ordinal variable"""
    merged_cat.LotShape.replace(to_replace = ['IR3', 'IR2', 'IR1', 'Reg'], value = [0, 1, 2, 3], inplace = True)
    merged_cat.LandContour.replace(to_replace = ['Low', 'Bnk', 'HLS', 'Lvl'], value = [0, 1, 2, 3], inplace = True)
    merged_cat.Utilities.replace(to_replace = ['NoSeWa', 'AllPub'], value = [0, 1], inplace = True)
    merged_cat.LandSlope.replace(to_replace = ['Sev', 'Mod', 'Gtl'], value = [0, 1, 2], inplace = True)
    merged_cat.ExterQual.replace(to_replace = ['Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3], inplace = True)
    merged_cat.ExterCond.replace(to_replace = ['Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4], inplace = True)
    merged_cat.BsmtQual.replace(to_replace = ['None', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4], inplace = True)
    merged_cat.BsmtCond.replace(to_replace = ['None', 'Po', 'Fa', 'TA', 'Gd'], value = [0, 1, 2, 3, 4], inplace = True)
    merged_cat.BsmtExposure.replace(to_replace = ['None', 'No', 'Mn', 'Av', 'Gd'], value = [0, 1, 2, 3, 4], inplace = True)
    merged_cat.BsmtFinType1.replace(to_replace = ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], value = [0, 1, 2, 3, 4, 5, 6], inplace = True)
    merged_cat.BsmtFinType2.replace(to_replace = ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], value = [0, 1, 2, 3, 4, 5, 6], inplace = True)
    merged_cat.HeatingQC.replace(to_replace = ['Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4], inplace = True)
    merged_cat.Electrical.replace(to_replace = ['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr'], value = [0, 1, 2, 3, 4], inplace = True)
    merged_cat.KitchenQual.replace(to_replace = ['Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3], inplace = True)
    merged_cat.Functional.replace(to_replace = ['Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'], value = [0, 1, 2, 3, 4, 5, 6], inplace = True)
    merged_cat.FireplaceQu.replace(to_replace =  ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4, 5], inplace = True)
    merged_cat.GarageFinish.replace(to_replace =  ['None', 'Unf', 'RFn', 'Fin'], value = [0, 1, 2, 3], inplace = True)
    merged_cat.GarageQual.replace(to_replace =  ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4, 5], inplace = True)
    merged_cat.GarageCond.replace(to_replace =  ['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], value = [0, 1, 2, 3, 4, 5], inplace = True)
    merged_cat.PavedDrive.replace(to_replace =  ['N', 'P', 'Y'], value = [0, 1, 2], inplace = True)
    merged_cat.PoolQC.replace(to_replace =  ['None', 'Fa', 'Gd', 'Ex'], value = [0, 1, 2, 3], inplace = True)
    merged_cat.Fence.replace(to_replace =  ['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'], value = [0, 1, 2, 3, 4], inplace = True)
    
    merged_cat.loc[:, ['OverallQual', 'OverallCond']] = merged_cat.loc[:, ['OverallQual', 'OverallCond']].astype('int64')
    merged_label_encoded = merged_cat.select_dtypes(include = ['int64'])
    
    merged_one_hot = merged_cat.select_dtypes(include=['category'])
    merged_one_hot = pd.get_dummies(merged_one_hot, drop_first=True)
    
    merged_encoded = pd.concat([merged_one_hot, merged_label_encoded], axis=1)
    merged_processed = pd.concat([merged_num_scaled, merged_encoded], axis=1)
    
    train_final = merged_processed.iloc[:train.shape[0], :]
    test_final = merged_processed.iloc[train.shape[0]:, :]
    
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir) 
    utils.save_as_pickle(train_final, os.path.join(output_dir, 'train.pkl'))
    utils.save_as_pickle(test_final, os.path.join(output_dir, 'test.pkl'))
    utils.save_as_pickle(train_target, os.path.join(output_dir, 'train_target.pkl'))
