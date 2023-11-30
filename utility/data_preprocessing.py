import pandas as pd
import re
import pickle
import lightgbm as lgb
from sklearn import preprocessing
from config.configuration import *
from sklearn.model_selection import KFold 
def preprocess_and_save(df):
  #remove null value from LossAmount
  df = df[pd.to_numeric(df['LossAmount'], errors='coerce').notna()]
 
  def extract_amount(value):
      match = re.search(r'(\d+\.\d+)', str(value))
      if match:
          return float(match.group(1))
      else:
          return None

  df['LossAmount'] = df['LossAmount'].apply(extract_amount)
  df = df.dropna(subset=['LossAmount'])
  df['LossAmount'].fillna(0, inplace=True)  
  additional_rows = df.sample(n=num_additional_rows, replace=True, random_state=42)
  additional_rows.reset_index(drop=True, inplace=True)
  df_processed = pd.concat([df, additional_rows], ignore_index=True)
  df_encoded = df_processed
#   encoder = preprocessing.LabelEncoder()
#   for col in columns_to_convert:
#         df_encoded[col]= encoder.fit_transform(df_encoded[col])
    # Initialize LabelEncoders
  label_encoders = {}
  for col in columns_to_convert:
       label_encoders[col] = preprocessing.LabelEncoder()
       df_encoded[col] = label_encoders[col].fit_transform(df_encoded[col])

  print(label_encoders)
  with open('./data/training/output/encoder.pkl', 'wb') as file:
       pickle.dump(label_encoders, file)
    
  #df_processed[columns_to_string] = df_processed[columns_to_string].astype(str)
  display(df_encoded)
  df_encoded.fillna(0, inplace=True)
  #df_processed[columns_to_int] = df_processed[columns_to_int].astype('int')
  df_encoded['Limit_val'] = df_encoded['Limit_val'].astype('double')
#df['YearBuilt'] = df['YearBuilt'].astype('double')
  df_encoded
  X = df_encoded.drop(columns=["LossAmount"])
  y = df_encoded["LossAmount"]
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler()
  scaler.fit(X_train)
  scaler.fit(X_test)
  return(X_train, y_train, X_test, y_test)