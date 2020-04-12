import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.pandas.set_option('display.max_columns', None)

dataset = pd.read_csv('train.csv')
dataset.head()

# Splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset, dataset['SalePrice'], 
                                                    test_size=0.1, random_state=1)
print(X_train.shape, X_test.shape)

# Missing Values in categorical features
features_nan = [feature for feature in dataset.columns if dataset[feature].isnull().sum() > 1 and dataset[feature].dtypes=='O']

for feature in features_nan:
    print('{}: {}% missing values'.format(feature, np.round(dataset[feature].isnull().mean(), 4)))

def replace_missing(dataset, features_nan):
    data = dataset.copy()
    data[features_nan] = data[features_nan].fillna('Missing')
    return data

dataset = replace_missing(dataset, features_nan)

# Missing Values in numerical features
numerical_with_nan_features = [feature for feature in dataset.columns if dataset[feature].isnull().sum() > 1 and dataset[feature].dtypes!='O']

for feature in numerical_with_nan_features:
    print('{}: {}% missing values'.format(feature, np.round(dataset[feature].isnull().mean(), 4)))

for feature in numerical_with_nan_features:
    median_value = dataset[feature].median()
    dataset[feature+'nan'] = np.where(dataset[feature].isnull(), 1, 0)
    dataset[feature].fillna(median_value, inplace=True)

dataset[numerical_with_nan_features].isnull().sum()
    
    
    
    
    
    