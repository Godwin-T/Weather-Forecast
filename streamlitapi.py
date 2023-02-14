import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import streamlit as st

path = ('C:/Users/Godwin/Documents/Workflow/Project/model.bin')

with open(path, 'rb') as f:
    model, scaler, dv = pickle.load(f)

def data_pred(df):

    num_col = ['rainfall', 'evaporation', 'sunshine',
                'windgustspeed', 'windspeed9am', 'windspeed3pm', 'humidity9am', 
                'humidity3pm', 'pressure9am', 'pressure3pm', 'cloud9am', 'cloud3pm', 
                 'temp9am']
                
    df['date'] = pd.to_datetime(df['date'])
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

def main():
    st.title('Rainfall Prediction API')

    #input variables
    df = {}
    date = st.text_input('date')
    location = st.text_input('location')
    rainfall = st.number_input('rainfall')
    evaporation = st.number_input('evaporation')
    sunshine = st.number_input('sunshine')
    windgustdir = st.text_input('windgustdir')
    windgustspeed = st.number_input('windgustspeed')
    winddir9am = st.text_input('winddir9am')
    winddir3pm = st.text_input('winddir3pm')
    windspeed9am = st.number_input('windspeed9am')
    windspeed3pm = st.number_input('windspeed3pm')
    humidity9am = st.number_input('humidity9am')
    humidity3pm = st.number_input('humidity3pm')
    pressure9am = st.number_input('pressure9am')
    pressure3pm = st.number_input('pressure3pm')
    cloud9am = st.number_input('cloud9am')
    cloud3pm = st.number_input('cloud3pm')
    temp9am = st.number_input('temp9am')
    raintoday = st.text_input('raintoday')

    keys = ['date', 'location', 'rainfall', 'evaporation', 'sunshine','windgustdir', 'windspeed9am',
            'windspeed3pm','windgustspeed', 'winddir9am', 'winddir3pm', 'humidity9am',
            'humidity3pm', 'pressure9am', 'pressure3pm', 'cloud9am',
            'cloud3pm', 'temp9am', 'raintoday']
    values = [date, location, rainfall, evaporation, sunshine, windgustdir, windspeed9am,
            windspeed3pm, windgustspeed, winddir9am, winddir3pm, humidity9am,
            humidity3pm, pressure9am, pressure3pm, cloud9am,
            cloud3pm, temp9am, raintoday]

    for i,v in zip(keys, values):
        df[i] = v
    

    if st.button('Predict'):
        data = data_pred(df)
        data = xgb.DMatrix(data)
        prediction = model.predict(data)
        if prediction < 0.5:
            output = "There won't be rain tomorrow"
        else:
            output = "It will rain tomorrow"

        st.success(output)

if __name__ == "__main__":
    main()

