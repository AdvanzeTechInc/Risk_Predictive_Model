import pandas as pd
import re
import pickle
import lightgbm as lgb
from datetime import datetime
from sklearn import preprocessing
from config.configuration import *
from sklearn.model_selection import KFold 
def preprocess_and_save(df):
  today = datetime.today().strftime('%Y%m%d')
  #remove null value from LossAmount
  df = df[pd.to_numeric(df['LossAmount'], errors='coerce').notna()]
  def extract_amount(value):
      match = re.search(r'(\d+\.\d+)', str(value))
      if match:
          return float(match.group(1))
      else:
          return None

  df['LossAmount'] = df['LossAmount'].apply(extract_amount)
  #df.dropna(subset=['LossAmount'], inplace=True)
#   df = df.dropna(subset=['LossAmount'])
#   df['LossAmount'].fillna(0, inplace=True) 
  additional_rows = df.sample(n=num_additional_rows, replace=True, random_state=42)
  additional_rows.reset_index(drop=True, inplace=True)
  df_processed = pd.concat([df, additional_rows], ignore_index=True)
  df_encoded = df_processed
  
#   modes = df_encoded[columns_to_string].mode().iloc[0]
#   modes_dict = modes.to_dict()
#   with open('./data/training/output/mode_values_dict_'+today+'.pkl', 'wb') as file:
#         pickle.dump(modes_dict, file)
#   df_encoded = df_encoded.apply(lambda col: col.fillna(modes_dict[col.name]) if col.name in columns_to_string else col)

  label_encoders = {}
  for col in columns_to_convert:
       label_encoders[col] = preprocessing.LabelEncoder()
       df_encoded[col] = label_encoders[col].fit_transform(df_encoded[col])

  print(label_encoders)
  with open('./data/training/output/encoder_'+today+'.pkl', 'wb') as file:
       pickle.dump(label_encoders, file)
    
#   modes = df_encoded[columns_to_string].mode().iloc[0]
#   modes_dict = modes.to_dict()
#   with open('./data/training/output/mode_values_dict_'+today+'.pkl', 'wb') as file:
#         pickle.dump(modes, file)
#   df_encoded = df_encoded.apply(lambda col: col.fillna(modes[col.name]) if col.name in columns_to_string else col)

  #df_processed[columns_to_string] = df_processed[columns_to_string].astype(str)
  display(df_encoded)
  df_encoded.fillna(0, inplace=True)
  #df_processed[columns_to_int] = df_processed[columns_to_int].astype('int')
  df_encoded['Limit_val'] = df_encoded['Limit_val'].astype('double')
  
  #df['YearBuilt'] = df['YearBuilt'].astype('double')
  #df_encoded = df_encoded.dropna(subset=['LossAmount'])
  #modes = df_encoded.mode().iloc[0]
  print(df_encoded.columns)
  #sting
   #pickle file
  # Fill NaN values in each column with the corresponding mode
  #df_encoded = df_encoded.apply(lambda col: col.fillna(modes[col.name])) 
  X = df_encoded.drop(columns=["LossAmount"])
  y = df_encoded["LossAmount"]
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  scaler.fit(X_train)
  scaler.fit(X_test)
  return(X_train, y_train, X_test, y_test)