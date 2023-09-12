
# =============================================================================
# Gradient Boosting Algorithm
# =============================================================================

# =============================================================================
# Problem Statement
# Using the Boston Housing Data, predict the prices using Gradient Boosting (XGBoost)
# =============================================================================


# =============================================================================
# Preparing the Enviornment
# =============================================================================

from sklearn.datasets import load_boston
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split


# =============================================================================
# Loading the Data
# =============================================================================

#import it from scikit-learn 
boston = load_boston()
print(boston.keys()) #boston variable itself is a dictionary, so you can check for its keys using the .keys() method.
print(boston.data.shape)
print(boston.feature_names)
print(boston.DESCR)


# =============================================================================
# Exploraotry Data Analysis
# =============================================================================

data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
data.head()
data['PRICE'] = boston.target #Dependent Variable
data.info()
stats_df = pd.DataFrame(data.describe())


#Separate the target variable and rest of the variables using .iloc to subset the data.
X, y = data.iloc[:,:-1],data.iloc[:,-1]

#XGBoost supports and gives it acclaimed performance and efficiency gains
data_dmatrix = xgb.DMatrix(data=X,label=y)

#Splitting the Data into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


#Fitting the XGBoost Model
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)


xg_reg.fit(X_train,y_train)

# Train Error
preds_train = xg_reg.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, preds_train))
print("TRAIN RMSE: %f" % (rmse_train))

#Test Error
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("TEST RMSE: %f" % (rmse))


#k-fold Cross Validation using XGBoost 
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

cv_results.head()


print((cv_results["test-rmse-mean"]).tail(1))


xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=50)


# =============================================================================
# Final Model
# =============================================================================


import matplotlib.pyplot as plt
xgb.plot_tree(xg_reg, num_trees=15)
plt.rcParams['figure.figsize']=('70','30')
plt.show()


# =============================================================================
# Feature Importance 
# =============================================================================

xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()
