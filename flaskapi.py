import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from flask import Flask, request, jsonify

path = ('C:/Users/Godwin/Documents/Workflow/Project/model.bin')

with open(path, 'rb') as f:
    model, scaler, dv = pickle.load(f)

def data_pred(df):

    num_col = ['rainfall', 'evaporation', 'sunshine',
                'windgustspeed', 'windspeed9am', 'windspeed3pm', 'humidity9am', 
                'humidity3pm', 'pressure9am', 'pressure3pm', 'cloud9am', 'cloud3pm', 
                 'temp9am']
                
    df['date'] = pd.to_datetime(df['date'])
    print(df['date'])
    df['year'] = df['date'].year
    df['month'] = df['date'].month
    df.pop('date')

    global_mean = scaler.mean_
    global_std = np.sqrt(scaler.var_)
    for i,col in enumerate(num_col):
        mean = global_mean[i]
        std = global_std[i]
        df[col] = (df[col] - mean)/std

    df = dv.transform([df])
    return df

app = Flask(__name__)

@app.route('/')

def home():
    return "Welcome to the rainfall prediction site"

@app.route('/predict', methods = ['GET'])

def predict():
    df = {}
    date = request.args.get('date')
    location = request.args.get('location')
    rainfall = request.args.get('rainfall')
    evaporation = request.args.get('evaporation')
    sunshine = request.args.get('sunshine')
    windgustdir = request.args.get('windgustdir')
    windgustspeed = request.args.get('windgustspeed')
    winddir9am = request.args.get('winddir9am')
    winddir3pm = request.args.get('winddir3pm')
    windspeed9am = request.args.get('windspeed9am')
    windspeed3pm = request.args.get('windspeed3pm')
    humidity9am = request.args.get('humidity9am')
    humidity3pm = request.args.get('humidity3pm')
    pressure9am = request.args.get('pressure9am')
    pressure3pm = request.args.get('pressure3pm')
    cloud9am = request.args.get('cloud9am')
    cloud3pm = request.args.get('cloud3pm')
    temp9am = request.args.get('temp9am')
    raintoday = request.args.get('raintoday')

    keys = ['date', 'location', 'rainfall', 'evaporation', 'sunshine','windgustdir', 'windspeed9am',
            'windspeed3pm','windgustspeed', 'winddir9am', 'winddir3pm', 'humidity9am',
            'humidity3pm', 'pressure9am', 'pressure3pm', 'cloud9am',
            'cloud3pm', 'temp9am', 'raintoday']
    values = [date, location, float(rainfall),  float(evaporation),  float(sunshine), windgustdir,  float(windspeed9am),
            float(windspeed3pm), float(windgustspeed), winddir9am, winddir3pm, float(humidity9am),
            float(humidity3pm), float(pressure9am), float(pressure3pm), float(cloud9am),
            float(cloud3pm), float(temp9am), raintoday]

    for i,v in zip(keys, values):
        df[i] = v
    
    data = data_pred(df)
    data = xgb.DMatrix(data)
    prediction = model.predict(data)
    if prediction < 0.5:
        output = "The won't be rain tomorrow"
    else:
        output = "It will rain tomorrow"
    return output



if __name__ == "__main__":
    app.run(debug=True)