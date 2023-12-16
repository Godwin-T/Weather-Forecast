# Importing Libraries
print("Importing Libraries")
import pandas as pd
import pickle
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score,f1_score
from sklearn.metrics import precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')


def load_data(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.lower().str.replace(' ', '_')
    cat_col = data.dtypes[data.dtypes == 'object'].index.tolist()

    for col in cat_col:
        data[col] = data[col].str.lower().str.replace(' ', '_')
    return data

def feature_engineering(df):
   
    df['temp_diff'] = df['maxtemp'] - df['mintemp']
    df['humidity_diff'] = df['humidity3pm'] - df['humidity9am']
    return df

def data_preparation(data):
     
    data['date'] = pd.to_datetime(data['date'])
    num_col = data.dtypes[data.dtypes == 'float64'].index.tolist()
    cat_col = data.dtypes[data.dtypes == 'object'].index.tolist()

    for col in num_col:
            data[col] = data[col].interpolate(method='linear')
    for col in num_col:
            data[col] = data[col].fillna(data[col].mean())

    for col in cat_col:
            data[col] = data[col].fillna(data[col].mode()[0])

    return data

def feature_selection(data):
     
     features = ['rainfall', 'sunshine', 'windgustspeed', 'pressure3pm', 'cloud3pm',
            'humidity3pm', 'temp_diff', 'raintoday', 'location', 'humidity_diff', 'raintomorrow']
     data = feature_engineering(data)
     data = data[features]
     return data

def split_data(df):


    locations = [x for x in df.location.unique()]

    full_train_df = pd.DataFrame(columns=df.columns)
    full_val_df = pd.DataFrame(columns=df.columns)

    for i in locations:
        curr_data = df[df.location == i]
        no_datapoint = curr_data.shape[0]
        train_size = int(0.8 * no_datapoint)
        training_data, val_data = curr_data[:train_size], curr_data[train_size:]
        full_train_df = pd.concat([full_train_df, training_data])
        full_val_df = pd.concat([full_val_df, val_data])

    train_data = full_train_df.reset_index(drop = True)
    val_data = full_val_df.reset_index(drop = True)
    
    y_train = (train_data['raintomorrow'] == 'yes').astype('int')
    y_val = (val_data['raintomorrow'] == 'yes').astype('int')

    del train_data['raintomorrow']
    del val_data['raintomorrow']

    return train_data, val_data,y_train, y_val

def encoding(train_data, val_data):

    features = [x for x in train_data.columns]
    text = 'Feature Names: '
    for col in features:
        text = text + col +', '
        
    cat_col = train_data.dtypes[train_data.dtypes == 'object'].index.tolist()
    num_col = train_data.dtypes[train_data.dtypes == 'float64'].index.tolist()

    dv = DictVectorizer(sparse = False)
    dv.fit(train_data[cat_col + num_col].to_dict(orient = 'records'))
    feature_names = dv.get_feature_names_out()

    

    X_train = dv.transform(train_data[cat_col + num_col].to_dict(orient = 'records'))
    X_val = dv.transform(val_data[cat_col + num_col].to_dict(orient = 'records'))

    return X_train, X_val, text, dv

def xg_boost(x_train,y_train,x_val,y_val):

    dtrain = xgb.DMatrix(x_train, label = y_train)
    dval = xgb.DMatrix(x_val, label = y_val)

    x_params = {
    'eta': 0.3,
    'max_depth':6,
    'min_child_weight':1,

    'objective':'binary:logistic',
    'n_threads':5,

    'seed':1,
    'verbosity':0
    }

    model = xgb.train(x_params, dtrain = dtrain,num_boost_round = 10)

    pred = model.predict(dval)
    auc = roc_auc_score(y_val,pred)

    out = (pred >= 0.5).astype('int')
    acc =  accuracy_score(y_val, out)
    precision = precision_score(y_val,out)
    recall = recall_score(y_val, out)
    f1 = f1_score(y_val, out)

    return acc, auc, precision,recall,f1, model

def training(data):
    train_data, val_data,y_train, y_val = split_data(data)
    X_train, X_val, feature_names, dv  = encoding(train_data, val_data)
    output = xg_boost(X_train, y_train, X_val, y_val)
    return output

def model(X_train, y_train, X_val, y_val):
    
    oversample = SMOTE()
    X_train, X_val, feature_names, dv = encoding(X_train, X_val)
    X_train, y = oversample.fit_resample(X_train, y_train)
    output = xg_boost(X_train, y, X_val, y_val)
    return output, dv

def saving_model(model, data_encoder):

    model_name = 'weather.bin'
    with open(model_name, 'wb') as f:
        pickle.dump([model, data_encoder], f)
    print("Successfully saved the model")

def run(path = './data/weatherAUS.csv'):
    
    print('Loading Data....')
    data = load_data(path)
    print("Data Preparation")
    data = data_preparation(data)
    print('Feature Selection')
    data = feature_selection(data)
    print("Spliting Data")
    train_data, val_data,y_train, y_val = split_data(data)
    print('Modeling')

    output, dv = model(train_data,y_train, val_data, y_val)
    (acc, auc, precision,recall,f1, output_model) = output
    print(f'Accuracy Score: {acc}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
    print("Saving Model")
    saving_model(output_model, dv)

run()