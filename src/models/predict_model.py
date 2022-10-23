import os
import time
import typing as tp
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from catboost import CatBoostRegressor, Pool

from src import utils

model_path = '../models'

RS = 35

metrics = [r2_score, mean_squared_error]


def get_metrics(y_true: np.ndarray, y_pred: np.ndarray, metrics: tp.List[tp.Callable] = metrics) -> str:
    return ', '.join([f'{i.__name__}={i(y_true, y_pred):.4f}' for i in metrics])


def evaluate(train, target, model, name) -> None:
    
    train_data, val_data, train_target, val_target = train_test_split(train, target, train_size=0.8, random_state=RS)

    model.fit(train_data, train_target)
    y_pred = model.predict(val_data)
    print(f'{name}: {get_metrics(val_target.to_numpy(), y_pred)}')
