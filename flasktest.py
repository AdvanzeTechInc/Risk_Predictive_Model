import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template
import re
import math
from utility.data_cleaning import *
from utility.data_preprocessing import *
#from utility.model_building import *
from config.configuration import *





app = Flask("__name__")

q = ""

@app.route("/")
def loadPage():
    return render_template('home.html')



@app.route("/flask_test", methods=['POST'])
def RiskPrediction():
    
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    inputQuery8 = request.form['query8']
    inputQuery9 = request.form['query9']
    inputQuery10 = request.form['query10']
    inputQuery11 = request.form['query11']
    inputQuery12 = request.form['query12']
    inputQuery13 = request.form['query13']
   
       
   

    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5,inputQuery6, inputQuery7, inputQuery8, inputQuery9, inputQuery10,inputQuery11, inputQuery12, inputQuery13]]
 
    # Create the pandas DataFrame 
    df = pd.DataFrame(data, columns = ['Jurisdiction', 'CoverageType', 'Deductible', 'Limit', 'InjuryCause','YearBuilt', 'Stories', 'Comments', 'ConstructionTypeDesc', 'ZipCode','E2Value', 'SquareFootage', 'BuildingLimit'])
 
    #import model and encoder  
    with open('./data/training/output/lgb_model.pkl', 'rb') as file:
        model=pickle.load(file)
    with open('./data/training/output/encoder.pkl', 'rb') as file:
        label_encoders=pickle.load(file)
    df_clean = clean_data(df)
    for col in label_encoders:
        df_clean[col] = label_encoders[col].transform(df_clean[col])
    predicted_loss_amount= model.predict(df_clean)
    output = predicted_loss_amount[0]

    
    
    return render_template('home.html', Predicted_loss_amount=output, query1 = request.form['query1'], query2 = request.form['query2'],query3 = request.form['query3'],query4 = request.form['query4'],query5 = request.form['query5'],query6 = request.form['query6'],query7 = request.form['query7'],query8 = request.form['query8'],query9 = request.form['query9'],query10 = request.form['query10'],query11 = request.form['query11'],query12 = request.form['query12'],query13 = request.form['query13'])
    

if __name__ == '__main__':
    app.run(debug=True)

