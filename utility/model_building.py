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
import pandas as pd
import pickle
from config.configuration import *
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from lightgbm import Dataset
from sklearn.ensemble import VotingRegressor
def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    today = datetime.today().strftime('%Y%m%d')
    #linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    R2= r2_score(y_test, y_pred)
    print('Linear Regression')
    print(rmse)
    print(R2)
    with open('./data/training/output/lr_model_'+today+'.pkl', 'wb') as file:
        pickle.dump(model, file)
   #xgboost
    xgr = XGBRegressor()
    xgr.fit(X_train, y_train)
    y_pred1 = xgr.predict(X_test)
    mse1 = mean_squared_error(y_test, y_pred1)
    rmse1 = np.sqrt(mse1)
    R2_1= r2_score(y_test, y_pred1)
    print('XGBoost')
    print(rmse1)
    print(R2_1)
    with open('./data/training/output/xgboost_model_'+today+'.pkl', 'wb') as file:
        pickle.dump(xgr, file)
    #random forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred2 = rf_model.predict(X_test)
    mse2 = mean_squared_error(y_test, y_pred2)
    rmse2 = np.sqrt(mse2)
    R2_2= r2_score(y_test, y_pred2)
    print('RF')
    print(rmse2)
    print(R2_2)
    with open('./data/training/output/rf_model_'+today+'.pkl', 'wb') as file:
        pickle.dump(rf_model, file)
    #LightGBM 
#     lgb = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
#     train_data = lgb.Dataset(X_train, label=y_train)
#     test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
#     lgb_model = lgb.train(params, train_data, valid_sets=[test_data])
#     y_pred3 = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
#     mse3 = mean_squared_error(y_test, y_pred3)
#     rmse3 = np.sqrt(mse3)
#     R2_3 = r2_score(y_test, y_pred3)
#     print('LBM')
#     print(rmse3)
#     print(R2_3)
#     with open('./data/training/output/lgb_model_'+today+'.pkl', 'wb') as file:
#         pickle.dump(lgb_model, file)
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
    print('CATBoost')
    print(rmse4)
    print(R2_4)
    with open('./data/training/output/catboost_'+today+'.pkl', 'wb') as file:
        pickle.dump(catboost, file)
    #lbm_kfold
    # Convert object columns to numeric
#     lgb = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
    X_train[columns_to_convert] = X_train[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    X_test[columns_to_convert] = X_test[columns_to_convert].apply(pd.to_numeric, errors='coerce')
  
    for train_idx, val_idx in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
#         train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
#         valid_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)

#     lgb_model1 = lgb.train(params, train_data, valid_sets=[valid_data], num_boost_round=1000)
#     y_pre5 = lgb_model1.predict(X_val_fold)
#     mse5 = mean_squared_error(y_val_fold, y_pred5)
#     rmse5 = np.sqrt(mse5)
#     R2_5= r2_score(y_val_fold, y_pred5)
#     print('LGM_Kfold')
#     print(rmse5)
#     print(R2_5)
#     with open('./data/training/output/lightgbm_kfold.pkl', 'wb') as file:
#        pickle.dump(lgb_model1, file)
    cbr = CatBoostRegressor(iterations=100,
                        depth=5,
                        learning_rate=0.01,
                        loss_function='RMSE',
                        verbose=0)
    cbr.fit(X_train_fold, y_train_fold)
    y_pred6 = cbr.predict(X_val_fold)
    mse6 = mean_squared_error(y_val_fold, y_pred6)
    rmse6 = np.sqrt(mse6)
    R2_6= r2_score(y_val_fold, y_pred6)
    print('cbr_Kfold')
    print(rmse6)
    print(R2_6)
    with open('./data/training/output/cbr_kfold.pkl', 'wb') as file:
       pickle.dump(cbr, file)
    xgr = XGBRegressor()
    xgr.fit(X_train_fold, y_train_fold)
    y_pred7 = xgr.predict(X_val_fold)
    mse7 = mean_squared_error(y_val_fold, y_pred7)
    rmse7 = np.sqrt(mse7)
    R2_7= r2_score(y_val_fold, y_pred7)
    print('xgb_Kfold')
    print(rmse7)
    print(R2_7)
    with open('./data/training/output/xgb_Kfold.pkl', 'wb') as file:
       pickle.dump(xgr, file)
    rf_model.fit(X_train_fold, y_train_fold)
    y_pred8 = rf_model.predict(X_val_fold)
    mse8 = mean_squared_error(y_val_fold, y_pred8)
    rmse8 = np.sqrt(mse8)
    R2_8= r2_score(y_val_fold, y_pred8)
    print(' rf_model')
    print(rmse8)
    print(R2_8)
    with open('./data/training/output/rf_Kfold.pkl', 'wb') as file:
        pickle.dump(rf_model, file)
    ensemble = VotingRegressor(estimators=[('xgb', xgr), ('catboost', cbr)], weights=[0.5, 0.5])
    ensemble.fit(X_train_fold, y_train_fold)
    y_pred9 = ensemble.predict(X_val_fold)
    mse9 = mean_squared_error(y_val_fold, y_pred9)
    rmse9 = np.sqrt(mse9)
    R2_9= r2_score(y_val_fold, y_pred9)
    print('ensemble')
    print(rmse9)
    print(R2_9)
    with open('./data/training/output/xgb_cbr_Kfold.pkl', 'wb') as file:
        pickle.dump(ensemble, file)
#     lgb = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
#     ensemble = VotingRegressor(estimators=[('xgb', xgr), ('lightgbm', lgb)], weights=[0.5, 0.5])
#     ensemble.fit(X_train_fold, y_train_fold)
#     y_pred10 = ensemble.predict(X_val_fold)
#     mse10 = mean_squared_error(y_val_fold, y_pred10)
#     rmse10 = np.sqrt(mse10)
#     R2_10= r2_score(y_val_fold, y_pred10)
#     print('xgb_lgb_Kfold')
#     print(rmse10)
#     print(R2_10)
#     with open('./data/training/output/xgb_lgb_Kfold.pkl', 'wb') as file:
#         pickle.dump(ensemble, file)
#     ensemble = VotingRegressor(estimators=[('lightgbm', lgb), ('catboost', cbr) ], weights=[ 0.3, 0.4])
#     ensemble.fit(X_train_fold, y_train_fold)
#     y_pred11 = ensemble.predict(X_val_fold)
#     mse11 = mean_squared_error(y_val_fold, y_pred11)
#     rmse11 = np.sqrt(mse11)
#     R2_11= r2_score(y_val_fold, y_pred11)
#     print('cbr_lgb_Kfold')
#     print(rmse11)
#     print(R2_11)
#     with open('./data/training/output/cbr_lgb_Kfold.pkl', 'wb') as file:
#         pickle.dump(ensemble, file)
#     ensemble = VotingRegressor(estimators=[('xgb', xgr), ('catboost', cbr), ('lightgbm', lgb)], weights=[0.3, 0.3, 0.4])
#     ensemble.fit(X_train_fold, y_train_fold)
#     y_pred12 = ensemble.predict(X_val_fold)
#     mse12 = mean_squared_error(y_val_fold, y_pred12)
#     rmse12 = np.sqrt(mse11)
#     R2_12= r2_score(y_val_fold, y_pred12)
#     print('xgb_cbr_lgb_Kfold')
#     print(rmse12)
#     print(R2_12)
#     with open('./data/training/output/xgb_cbr_lgb_Kfold.pkl', 'wb') as file:
#         pickle.dump(ensemble, file)
    return rmse2, R2_2
