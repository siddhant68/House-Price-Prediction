import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

pd.pandas.set_option('display.max_columns', None)

dataset = pd.read_csv('X_train.csv')

y_train = dataset[['SalePrice']]

X_train = dataset.drop(['Id', 'SalePrice'], axis=1)

# Apply Feature Selection
feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0))
feature_sel_model.fit(X_train, y_train)

feature_sel_model.get_support()

selected_feat = X_train.columns[(feature_sel_model.get_support())]

print(len(selected_feat))