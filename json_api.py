import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template,jsonify
import re
import math
from utility.data_cleaning import *
from utility.data_preprocessing import *
from utility.model_building import *
from config.configuration import *
app = Flask(__name__)

#import model and encoder  
with open('./data/training/output/lgb_model.pkl', 'rb') as file:
    model=pickle.load(file)
with open('./data/training/output/encoder.pkl', 'rb') as file:
    label_encoders=pickle.load(file)


@app.route("/predict_api", methods=['POST'])
def RiskPrediction():
    
    json_ = request.json
    df= pd.DataFrame([json_])
    df_clean = clean_data(df)
    for col in label_encoders:
        df_clean[col] = label_encoders[col].transform(df_clean[col])
    predicted_loss_amount= model.predict(df_clean)
    output = predicted_loss_amount[0]

    print(output)
    
    return jsonify({"prediction":output})
if __name__ == '__main__':
    app.run()
