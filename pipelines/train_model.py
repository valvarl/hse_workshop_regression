import os
import numpy as np
import pandas as pd
import typing as tp

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor, Pool

from src import utils
from src.models import evaluate

RS = 35

model_path = '../models'


def train(input_train_data, input_train_target, output_model_filepath: str = model_path) -> None:

    train = pd.read_pickle(input_train_data)
    target = pd.read_pickle(input_train_target)
    
    train_data, val_data, train_target, val_target = train_test_split(train, target, train_size=0.8, random_state=RS)
    
    # Ridge

    parameters = {
        'fit_intercept': [True, False],
        'alpha': [0.1, 1, 5, 10, 15, 100],
        'tol': [1e-5, 1e-3, 1e-1],
        'positive': [True, False]
    }

    model = Ridge(random_state=RS, max_iter=1000)
    clf = GridSearchCV(model, parameters, scoring='r2', cv=3)
    clf.fit(train_data, train_target)
    clf.best_params_

    ridge = Ridge(random_state=RS, max_iter=1000, **clf.best_params_).fit(train_data, train_target)

    utils.save_model(ridge, os.path.join(output_model_filepath, 'ridge.pkl'))
    
    evaluate.evaluate(train, target, ridge, 'Ridge')
    
    # Catboost
    
    pool = Pool(train_data, train_target, )

    cb = CatBoostRegressor(iterations=2000, loss_function='RMSE', eval_metric='RMSE', learning_rate=0.03, silent=True)

    cb.fit(pool, eval_set=(val_data, val_target), verbose=False, plot=False)
    
    utils.save_model(cb, os.path.join(output_model_filepath, 'catboost.pkl'))
    
    evaluate.evaluate(train, target, cb, 'Catboost')
