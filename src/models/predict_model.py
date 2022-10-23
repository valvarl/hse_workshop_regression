import os
import time
import numpy as np
import pandas as pd

from src import utils

data_path = 'data/processed'
model_path = 'models'
retort_path = 'reports'

train = pd.read_pickle(os.path.join(data_path, 'train.pkl'))
target = pd.read_pickle(os.path.join(data_path, 'train_target.pkl'))
test = pd.read_pickle(os.path.join(data_path, 'test.pkl'))

ridge = utils.load_model(os.path.join(model_path, 'ridge.pkl'))
catboost = utils.load_model(os.path.join(model_path, 'catboost.pkl'))

models = [ridge, catboost]

for name, model in zip(['ridge', 'catboost'], models):
    model.fit(train, target)
    y_pred = model.predict(test)
    res = pd.concat([pd.Series(test.index), pd.Series(np.expm1(y_pred))], axis=1)
    res.to_csv(os.path.join(retort_path, f'{name}_{time.time()}.csv'))
