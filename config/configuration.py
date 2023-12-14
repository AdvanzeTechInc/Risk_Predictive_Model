#def process_data(df):
    #keep only selected columns
#input_data_path="./data/training/input/Data for loss amount prediction.xlsx"  
input_data_path= "./data/training/input/ClaimDetailsnew.xlsx"
feature_columns = ['Jurisdiction', 'CoverageType', 'Deductible', 'Limit_val', 'InjuryCause','YearBuilt', 'Stories', 'Comments', 'ConstructionTypeDesc', 'ZipCode', 'E2Value', 'SquareFootage', 'BuildingLimit']
 
selected_columns = ['Jurisdiction', 'CoverageType', 'Deductible', 'Limit_val', 'InjuryCause','YearBuilt', 'Stories', 'Comments', 'ConstructionTypeDesc', 'ZipCode', 'LossAmount','E2Value', 'SquareFootage', 'BuildingLimit']
 
col_to_drop= [ "Deductible", "YearBuilt"]

num_additional_rows = 190
columns_to_string = ['Jurisdiction', 'CoverageType', 'InjuryCause', 'ClassCode', 'ConstructionTypeDesc', 'ZipCode','Deductible_type']
columns_to_int = ['Stories', 'Deductible_val','SquareFootage','BuildingLimit','YearBuilt']
columns_to_float=['E2Value','Limit_val']
columns_to_convert = ['Jurisdiction', 'CoverageType', 'InjuryCause', 'ClassCode', 'ConstructionTypeDesc', 'ZipCode', 'Deductible_type']
from datetime import datetime
today = datetime.today().strftime('%Y%m%d')
params = {
    'objective': 'regression',
    'metric': ['l1', 'l2'],
    'learning_rate': 0.005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.7,
    'bagging_freq': 10,
    'verbose': 0,
    'max_depth': 8,
    'num_leaves': 128,
    'max_bin': 512,
    'num_boost_round': 100000,
    'early_stopping_round': 500,
}
from sklearn.model_selection import KFold      
num_folds = 8
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

