import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.pandas.set_option('display.max_columns', None)

dataset_test = pd.read_csv('test.csv')

# Missing values in categorical features
categorical_features_with_nan = [feature for feature in dataset_test.columns if dataset_test[feature].dtype == 'O' and dataset_test[feature].isnull().sum()>1]

for feature in categorical_features_with_nan:
    dataset_test[feature] = dataset_test[feature].fillna('Missing')
    
# Missing values in numerical features
numerical_features_with_nan = [feature for feature in dataset_test.columns if dataset_test[feature].dtype != 'O' and dataset_test[feature].isnull().sum()>1]

for feature in numerical_features_with_nan:
    median_value = dataset_test[feature].median()
    dataset_test[feature+'nan'] = np.where(dataset_test[feature].isnull(), 1, 0)
    dataset_test[feature].fillna(median_value)

# Data Time Variables
for feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:    
    dataset_test[feature] = dataset_test['YrSold'] - dataset_test[feature]

## Converting skewed data in continuous features 
# to normally distributed data using logarithmic transformation
num_features = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']

for feature in num_features:
    dataset_test[feature] = np.log(dataset_test[feature])
    
# Handling rare categorical features
categorical_features = [feature for feature in dataset_test.columns if dataset_test[feature].dtype == 'O']
for feature in categorical_features:    
    temp = dataset_test.groupby(feature)['LotArea'].count()/ len(dataset_test)
    temp_df = temp[temp > 0.01].index
    dataset_test[feature] = np.where(dataset_test[feature].isin(temp_df), dataset_test[feature], 'Rare_var')

for feature in categorical_features:
    labels_ordered = dataset_test.groupby([feature])['LotArea'].mean().sort_values().index
    labels_ordered = {k: i for i, k in enumerate(labels_ordered,0)}
    dataset_test[feature] = dataset_test[feature].map(labels_ordered)

# Feature Scaling    
feature_scale = [feature for feature in dataset_test.columns if feature != 'Id']
    
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
    
    
    
    
    
    
    
    
    
    
    