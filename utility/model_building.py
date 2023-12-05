from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import xgboost
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import pickle
from config.configuration import *
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from lightgbm import Dataset

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    today = datetime.today().strftime('%Y%m%d')
    #linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    R2= r2_score(y_test, y_pred)
    with open('./data/training/output/lr_model_'+today+'.pkl', 'wb') as file:
        pickle.dump(model, file)
   #xgboost
    xgr = XGBRegressor()
    xgr.fit(X_train, y_train)
    y_pred1 = xgr.predict(X_test)
    mse1 = mean_squared_error(y_test, y_pred1)
    rmse1 = np.sqrt(mse1)
    R2_1= r2_score(y_test, y_pred1)
    with open('./data/training/output/xgboost_model_'+today+'.pkl', 'wb') as file:
        pickle.dump(xgr, file)
    #random forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred2 = rf_model.predict(X_test)
    mse2 = mean_squared_error(y_test, y_pred2)
    rmse2 = np.sqrt(mse2)
    R2_2= r2_score(y_test, y_pred2)
    with open('./data/training/output/rf_model_'+today+'.pkl', 'wb') as file:
        pickle.dump(rf_model, file)
    #LightGBM  
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    lgb_model = lgb.train(params, train_data, valid_sets=[test_data])
    y_pred3 = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    mse3 = mean_squared_error(y_test, y_pred3)
    rmse3 = np.sqrt(mse3)
    R2_3 = r2_score(y_test, y_pred3)
    with open('./data/training/output/lgb_model_'+today+'.pkl', 'wb') as file:
        pickle.dump(lgb_model, file)
    #catboost
    catboost = CatBoostRegressor(iterations=100,
                        depth=5,
                        learning_rate=0.01,
                        loss_function='RMSE',
                        verbose=0)
    catboost.fit(X_train, y_train)
    y_pred4 = catboost.predict(X_test)
    mse4 = mean_squared_error(y_test, y_pred4)
    rmse4 = np.sqrt(mse4)
    R2_4 = r2_score(y_test, y_pred4)
    with open('./data/training/output/catboost_'+today+'.pkl', 'wb') as file:
        pickle.dump(catboost, file)
    #lbm_kfold
#     # Convert object columns to numeric
#     X_train[columns_to_convert] = X_train[columns_to_convert].apply(pd.to_numeric, errors='coerce')
#     X_test[columns_to_convert] = X_test[columns_to_convert].apply(pd.to_numeric, errors='coerce')
  
#     for train_idx, val_idx in kf.split(X_train):
#         X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
#         y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
#         train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
#         valid_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

#     lgb_model1 = lgb.train(params, train_data, valid_sets=[valid_data], num_boost_round=1000)
#     y_pred5 = lgb_model1.predict(X_val_fold)
#     mse5 = mean_squared_error(y_val_fold, y_pred5)
#     rmse5 = np.sqrt(mse5)
#     R2_5= r2_score(y_val_fold, y_pred5)
#     with open('./data/training/output/lightgbm_kfold.pkl', 'wb') as file:
#        pickle.dump(lgb_model1, file)
    return rmse2, R2_2
