import requests

url = "http://127.0.0.1:5080/predict"

data = {'date': '2008-12-11',
        'location': 'albury',
        'mintemp': 13.4,
        'maxtemp': 30.4,
        'rainfall': 0.0,
        'evaporation': 6.5346375106698895,
        'sunshine': 7.649576073624031,
        'windgustdir': 'n',
        'windgustspeed': 30.0,
        'winddir9am': 'sse',
        'winddir3pm': 'ese',
        'windspeed9am': 17.0,
        'windspeed3pm': 6.0,
        'humidity9am': 48.0,
        'humidity3pm': 22.0,
        'pressure9am': 1011.8,
        'pressure3pm': 1008.7,
        'cloud9am': 6.6,
        'cloud3pm': 8.0,
        'temp9am': 20.4,
        'temp3pm': 28.8,
        'raintoday': 'no'}


response = requests.post(url, json = data).json()
print(response)

# if response['Churn'] == True:
#     print(f"The customer with the an ID of {data['customerid']} probability of churning is {response['Prediction']}")