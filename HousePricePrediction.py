import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

pd.pandas.set_option('display.max_columns', None)

dataset = pd.read_csv('train.csv')

dataset.head()

# ---------------------------------- FEATURE ENGINEERING --------------------------------------#


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

for feature in categorical_features:
    labels_ordered = dataset.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered = {k: i for i, k in enumerate(labels_ordered,0)}
    dataset[feature] = dataset[feature].map(labels_ordered)

# Apply Feature Selection
dataset = dataset.drop(['Id'], axis=1)
feature_sel_model = SelectFromModel(Lasso(alpha=0.005, random_state=0))
feature_sel_model.fit(dataset.iloc[:, :80].values, dataset.iloc[:, 80:81].values)

selected_feat = dataset.columns[list(feature_sel_model.get_support())+[True, True, True]]
dataset = dataset.loc[:, dataset.columns.intersection(list(selected_feat)+['SalePrice'])]

X = dataset.iloc[:, :38].values
y = dataset.iloc[:, 38:39].values

# Splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(X_train.shape, X_test.shape)

# Feature Scaling    
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ------------------------------------- MODELLING --------------------------------------------#


# Fitting XGBoost to dataset
import xgboost
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV

regressor_xg = xgboost.XGBRegressor()

# Hyper Parameter Optimization
n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]
base_score=[0.25,0.5,0.75,1]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'learning_rate': learning_rate,
    'min_child_weight': min_child_weight,
    'booster': booster,
    'base_score': base_score
    }

# Set up the random search with 4-fold cross validation
random_cv = RandomizedSearchCV(estimator = regressor_xg,
                param_distributions = hyperparameter_grid,
                cv = 5, n_iter = 50,
                scoring = 'neg_mean_absolute_error',n_jobs = 4,
                verbose = 5, 
                return_train_score = True,
                random_state = 42
            )

random_cv.fit(X_train, y_train)

random_cv.best_estimator_

regressor_xg = xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=1, gamma=0,
                 importance_type='gain', learning_rate=0.05, max_delta_step=0,
                 max_depth=2, min_child_weight=4, missing=None, n_estimators=900,
                 n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                 silent=None, subsample=1, verbosity=1
             )

regressor_xg.fit(X_train, y_train)

# Pickling
filename = 'house_pricing_model.sav'
pickle.dump(regressor_xg, open(filename, 'wb'))

# Loading pickled model
regressor_xg = pickle.load(open('house_pricing_model.sav', 'rb'))

y_pred_xg = regressor_xg.predict(X_test)
Rsquare_xg = regressor_xg.score(X_test, y_test)
accuracy_xg = metrics.explained_variance_score(y_test, y_pred_xg)
print(accuracy_xg)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor

regressor_rf = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor_rf.fit(X_train, y_train)

y_pred_rf = regressor_rf.predict(X_test)
Rsquare_rf = regressor_rf.score(X_test, y_test)
accuracy_rf = metrics.explained_variance_score(y_test, y_pred_rf)

print(format(accuracy_xg*100, '.2f'), format(accuracy_rf*100, '.2f'))

# Plotting results
plt.plot(y_pred_rf, color='red', label='Predicted House Prices')
plt.plot(y_test, color='green', label='Real House Prices')
plt.title('Real VS Predicted House Prices')
plt.xlabel('Houses')
plt.ylabel('Price')
plt.legend()
plt.plot()