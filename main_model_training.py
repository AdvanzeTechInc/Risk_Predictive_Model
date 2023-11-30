#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re
import xgboost
import lightgbm as lgb
import catboost
from sklearn.model_selection import KFold


# In[2]:


from utility.data_cleaning import *
from utility.data_preprocessing import *
from utility.model_building import *
from config.configuration import *


# In[4]:


from datetime import datetime
today = datetime.today()
print(today)


# In[3]:


df=pd.read_excel(input_data_path)
df


# In[4]:


df = df[selected_columns]
df_clean= clean_data(df)
df_clean


# In[5]:


X_train, y_train, X_test, y_test= preprocess_and_save(df_clean)


# In[6]:


train_and_evaluate_model(X_train, y_train, X_test, y_test)


# In[ ]:




