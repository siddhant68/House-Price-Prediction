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

# Date Time Variables    
for feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:    
    dataset[feature] = dataset['YrSold'] - dataset[feature]

dataset.head()

# Converting skewed data in continuous features 
# to normally distributed data using logarithmic transformation
num_features = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']

for feature in num_features:
    dataset[feature] = np.log(dataset[feature])

# Handling Rare Categorical Feature
categorical_features = [feature for feature in dataset.columns if dataset[feature].dtype == 'O']

print(len(categorical_features))

for feature in categorical_features:
    temp = dataset.groupby(feature)['SalePrice'].count()/len(dataset)
    temp_df = temp[temp > 0.01].index
    dataset[feature] = np.where(dataset[feature].isin(temp_df), dataset[feature], 'Rare_var')

# Feature Scaling
for feature in categorical_features:
    labels_ordered = dataset.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered = {k: i for i, k in enumerate(labels_ordered,0)}
    dataset[feature] = dataset[feature].map(labels_ordered)
    
feature_scale = [feature for feature in dataset.columns if feature not in ['Id', 'SalePrice']]
    
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(dataset[feature_scale])

# transform the train and test set, and add on the Id and SalePrice variables
data = pd.concat([dataset[['Id', 'SalePrice']].reset_index(drop=True),
                  pd.DataFrame(scaler.transform(dataset[feature_scale]), columns=feature_scale)],
                  axis=1)

data.to_csv('X_train.csv', index=False)








