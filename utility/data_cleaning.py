# Your utility code goes here
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re
from datetime import datetime
from config.configuration import *

def clean_data(df_processed):
  #lower case
  #df_processed = df_processed[selected_columns]
  df_processed['Comments'] = df_processed['Comments'].str.replace('ClassCode:', '')
  df_processed = df_processed.rename(columns={'Comments': 'ClassCode'})
  display(df_processed)
  df_processed["ZipCode"] = df_processed["ZipCode"].str.replace("-", "")
  df_processed['limit_type'] = df_processed['Limit'].str.extract(r'([a-zA-Z]+)')
  df_processed['Limit_val'] = df_processed['Limit'].str.extract(r'(\d+)')
  df_processed['Deductible_type'] = df_processed['Deductible'].str.extract(r'([a-zA-Z]+)')
  df_processed['Deductible_val'] = df_processed['Deductible'].str.extract(r'(\d+)')
  df_processed[columns_to_int] = df_processed[columns_to_int].astype('int')
  df_processed[columns_to_string] = df_processed[columns_to_string].astype('string')
  df_processed[columns_to_float] = df_processed[columns_to_float].astype('float')  
  df_processed['YearGap'] = datetime.now().year - df_processed['YearBuilt']
  df_processed=df_processed.drop(columns=col_to_drop)
  #df_processed=df_processed.dropna()
  return df_processed