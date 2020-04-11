import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.pandas.set_option('display.max_columns', None)

dataset = pd.read_csv('train.csv')

dataset.shape


# Missing Values
features_with_na = [feature for feature in dataset.columns if dataset[feature].isnull().sum()>1]

for feature in features_with_na:
    print(feature, np.round(dataset[feature].isnull().mean()*100, 4), '% missing values')

# Relationship of these columns with SalePrice
for feature in features_with_na:
    data = dataset.copy()
    data[feature] = np.where(data[feature].isnull(), 1, 0)
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()

# Numerical Values
numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']
print(len(numerical_features))

dataset[numerical_features].head()

# Year value in numerical features
year_features = [feature for feature in dataset.columns if 'Yr' in feature or 'Year' in feature]
print(len(year_features))

dataset.groupby('YrSold')['SalePrice'].median().plot()
plt.title('Year Sold vs SalePrice')
plt.show()

# Difference between Year Sold and other years VS Price
for feature in year_features:
    if feature != 'YrSold':
        data[feature] = dataset['YrSold'] - dataset[feature]
        plt.scatter(data[feature], dataset['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()

# Discrete Feature
discrete_features = [feature for feature in numerical_features if len(dataset[feature].unique()) < 25 and feature not in year_features + ['Id']]
print("Discrete Variables Count: {}".format(len(discrete_features)))

# Relationship of these features with SalePrice
for feature in discrete_features:
    data = dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
    
# Continuous Feature
continuous_features = [feature for feature in numerical_features if feature not in discrete_features + year_features + ['Id']]
print('Continuous Variables Count: {}'.format(len(continuous_features)))

# Relationship of these features with SalePrice
for feature in continuous_features:
    data = dataset.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.title(feature)
    plt.show()

# Converting skewed data in continuous features 
# to normally distributed data using logarithmic transformation
for feature in continuous_features:
    data = dataset.copy()
    if 0 in dataset[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data['SalePrice'] = np.log(data['SalePrice'])
        plt.scatter(data[feature], data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()
    
# Outliers
for feature in continuous_features:
    data = dataset.copy()
    if 0 in dataset[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()

# Categorical Features
categorical_features = [feature for feature in dataset.columns if dataset[feature].dtypes == 'O']
print('No of categorical features {}'.format(len(categorical_features)))

# No of categories in each categorical features
for feature in categorical_features:
    print('Feature {}, No of Categories {}'.format(feature, len(dataset[feature].unique())))

# Relationship of categorical features with dependent feature






