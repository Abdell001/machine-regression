# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 22:44:50 2021

@author: hp
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv')
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalised_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
features_df = normalised_df.drop(colums =['date', 'light'])
Appliances = normalised_df ['Appliances']
from sklearn.model import train_test_split
x_train, x_test, y_train, y_test = train_test_split (features_df, Appliances, teat_size = 70 - 30, random_state = 42)

from sklearn import linear_model
linear_model = linear_model.LinearRegression()
linear_model.fit(x_train, y_train)
predicted_values = linear_model.predict(x_test)
from sklearn.metrics import r2_score
r2_score = r2_score (y_test, predicted_values)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error (y_test, predicted_values)
round (mae, 3)
import numpy as np
rss = np.sum (np.square(y_test - predicted_values))
round (rss, 3)
from sklearn.martics import mean_squared_error
rmse = np.sqrt(mean_squared_error (y_test, predicted_values))
round (rmse, 3)
