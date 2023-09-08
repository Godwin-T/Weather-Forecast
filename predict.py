#Importing Libries
import pandas as pd
import pickle
import xgboost as xgb
from flask import Flask, request, jsonify


# Loading model
path = './weather.bin'
with open(path, 'rb') as f:
    model, dv = pickle.load(f)


def load_data(path):

    data = pd.read_csv(path)
    data.columns = data.columns.str.lower().str.replace(' ', '_')
    return data

def feature_engineering(df):
   
    df['temp_diff'] = df['maxtemp'] - df['mintemp']
    df['humidity_diff'] = df['humidity3pm'] - df['humidity9am']
    return df

def feature_selection(dict):

    features = ['rainfall', 'sunshine', 'windgustspeed', 'pressure3pm', 'cloud3pm',
            'humidity3pm', 'temp_diff', 'raintoday', 'location', 'humidity_diff']
    
    new_dict = {}
    for feat in features:
        new_dict[feat] = dict[feat]
    return new_dict

def encoding(data, dict_vec):

    data = dict_vec.transform(data)
    data = xgb.DMatrix(data)
    return data

def run(data):
    
    data = feature_engineering(data)
    data = feature_selection(data)
    data = encoding(data, dv)
    pred = model.predict(data)
    out = (pred >= 0.5).astype('int')
    return pred, out

app = Flask(__name__)
@app.route('/')

def home():
    return "Welcome to the rainfall prediction site"

@app.route("/predict", methods = ['POST'])
def predict():

    customer = request.get_json()
    pred, out = run(customer)

    results = {'Prediction':float(pred), 'Rain':bool(out)}
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True, port=5080)