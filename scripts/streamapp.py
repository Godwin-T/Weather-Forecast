import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import streamlit as st

# Loading model
path = '../models/model.pkl'

with open(path, 'rb') as f:
    model, scaler, vectorizer = pickle.load(f)

numerical_col = ['mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine',
                'windgustspeed', 'windspeed9am', 'windspeed3pm', 'humidity9am',
                'humidity3pm', 'pressure9am', 'pressure3pm', 'cloud9am',
                'cloud3pm', 'temp9am', 'temp3pm']

def prediction(data):

    numerical_values = [data[col] for col in numerical_col]
    numerical_values = scaler.transform([numerical_values])[0]
    for col, value in zip(numerical_col, numerical_values):
        data[col] =  value
    data = vectorizer.transform(data)
    data = xgb.DMatrix(data)
    prediction = model.predict(data)
    return prediction
def main():
    st.title('Rainfall Prediction API')
    
    #input variables
    dicts = {}
    location = st.selectbox('Location ', ('albury', 'badgeryscreek', 'cobar', 'coffsharbour', 'moree',
                                        'newcastle', 'norahhead', 'norfolkisland', 'penrith', 'richmond',
                                        'sydney', 'sydneyairport', 'waggawagga', 'williamtown',
                                        'wollongong', 'canberra', 'tuggeranong', 'mountginini', 'ballarat',
                                        'bendigo', 'sale', 'melbourneairport', 'melbourne', 'mildura',
                                        'nhil', 'portland', 'watsonia', 'dartmoor', 'brisbane', 'cairns',
                                        'goldcoast', 'townsville', 'adelaide', 'mountgambier', 'nuriootpa',
                                        'woomera', 'albany', 'witchcliffe', 'pearceraaf', 'perthairport',
                                        'perth', 'salmongums', 'walpole', 'hobart', 'launceston',
                                        'alicesprings', 'darwin', 'katherine', 'uluru'))
    
    mintemp = st.number_input('Minimum Temperature')
    maxtemp = st.number_input('Maximum Temperature')
    rainfall = st.number_input('Rainfall Intensity')
    evaporation = st.number_input('Evaporation Level')
    sunshine = st.number_input('Sunshine Intensity')

    windgustdir = st.selectbox('Wind Direction',('w', 'wnw', 'wsw', 'ne', 'nnw', 'n', 'nne', 'sw', 'ene', 
                                                 'sse', 's', 'nw', 'se', 'ese', 'e', 'ssw'))
    
    windgustspeed = st.number_input('Wind Speed')
    winddir9am = st.selectbox('9am Wind Direction', ('w', 'nnw', 'se', 'ene', 'sw', 'sse', 's', 'ne', 'n', 
                                                     'ssw', 'wsw', 'ese', 'e', 'nw', 'wnw', 'nne'))
    winddir3pm = st.selectbox('3pm Wind Direction', ('wnw', 'wsw', 'e', 'nw', 'w', 'sse', 'ese', 'ene', 'nnw', 
                                                      'ssw', 'sw', 'se', 'n', 's', 'nne', 'ne'))
    
    windspeed9am = st.number_input('9am Wind Speed')
    windspeed3pm = st.number_input('3pm Wind Speed')
    humidity9am = st.number_input('9am Humidity Level')
    humidity3pm = st.number_input('3pm Humidity Level')
    pressure9am = st.number_input('9am Pressure Level')
    pressure3pm = st.number_input('3pm Pressure Level')
    cloud9am = st.number_input('9am Cloud Level')
    cloud3pm = st.number_input('3pm Cloud Level')
    temp9am = st.number_input('9am Temperatue Level')
    temp3pm = st.number_input('3pm Temperatue Level')
    raintoday = st.selectbox('Did it rain today', ('no', 'yes'))

    keys = ['location', 'mintemp', 'maxtemp', 'rainfall', 'evaporation', 'sunshine',
            'windgustdir', 'windgustspeed', 'winddir9am', 'winddir3pm',
            'windspeed9am', 'windspeed3pm', 'humidity9am', 'humidity3pm',
            'pressure9am', 'pressure3pm', 'cloud9am', 'cloud3pm', 'temp9am',
            'temp3pm', 'raintoday']
    
    values = [location, mintemp, maxtemp, rainfall, evaporation, sunshine,
               windgustdir, windgustspeed, winddir9am, winddir3pm, windspeed9am,
               windspeed3pm,humidity9am, humidity3pm, pressure9am, pressure3pm, 
               cloud9am, cloud3pm, temp9am, temp3pm, raintoday]

    for key,value in zip(keys, values):
        dicts[key] = value
    

    if st.button('Predict'):
        pred = prediction(dicts)

        if pred < 0.5:
            output = "There won't be rain tomorrow"
        else:
            output = "It will rain tomorrow"

        st.success(output)

if __name__ == "__main__":
    main()

