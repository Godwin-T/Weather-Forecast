print('Importing Libraries')
#importing libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')
print('Completed')
out = 'model.pkl'

#Loading Data
def load_data(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.lower().str.replace(' ', '_')
    cat_col = data.dtypes[data.dtypes == 'object'].index.tolist()

    for col in cat_col:
        data[col] = data[col].str.lower().str.replace(' ', '_')
    return data

#Spliting Data
def split_data(df):
        
    full_train_df, full_test_df = train_test_split(df, test_size = 0.2, random_state=1)
    full_train_df, full_val_df = train_test_split(full_train_df, test_size = 0.25, random_state=1)

    train_data = full_train_df.reset_index(drop = True)
    test_data = full_test_df.reset_index(drop = True)
    val_data = full_val_df.reset_index(drop = True)

    y_train = (train_data['raintomorrow'] == 'yes').astype('int')
    y_test = (test_data['raintomorrow'] == 'yes').astype('int')
    y_val = (val_data['raintomorrow'] == 'yes').astype('int')

    del train_data['raintomorrow']
    del test_data['raintomorrow']
    del val_data['raintomorrow']

    return train_data, val_data,test_data, y_train, y_val, y_test,full_train_df

#Encoding Data
def encoding(train_data, test_data):
    cat_col = train_data.dtypes[train_data.dtypes == 'object'].index.tolist()
    num_col = train_data.dtypes[train_data.dtypes == 'float64'].index.tolist()

    dv = DictVectorizer(sparse = False)
    dv.fit(train_data[cat_col + num_col].to_dict(orient = 'records'))
    feature_names = dv.get_feature_names()

    X_train = dv.transform(train_data[cat_col + num_col].to_dict(orient = 'records'))
    X_test = dv.transform(test_data[cat_col + num_col].to_dict(orient = 'records'))

    return dv,X_train,X_test

#Defining xgboost model
def xg_boost(x_train,y_train,x_test, y_test):

    dtrain = xgb.DMatrix(x_train, label = y_train)
    dtest = xgb.DMatrix(x_test)

    x_params = {
    'eta': 0.3,
    'max_depth':6,
    'min_child_weight':1,

    'objective':'binary:logistic',
    'eval_metric' : 'map',
    
    'n_threads':8,
    'seed':1,
    'verbosity':0
    }

    model = xgb.train(x_params, dtrain = dtrain,num_boost_round = 25)

    pred = model.predict(dtest)
    out = (pred >= 0.5).astype('int')
    precision = precision_score(y_test,out)
    recall = recall_score(y_test, out)
    f1 = f1_score(y_test, out)

    return (precision,recall,f1, model)

#Data Preparation
def data_prep(df):

    #filling missing values
    df['date'] = pd.to_datetime(df['date'])

    num_col = df.dtypes[df.dtypes == 'float64'].index.tolist()
    cat_col = df.dtypes[df.dtypes == 'object'].index.tolist()

    for col in num_col:
            df[col] = df[col].interpolate(method='linear')
    for col in num_col:
            df[col] = df[col].fillna(df[col].mean())

    for col in cat_col:
            df[col] = df[col].fillna(df[col].mode()[0])

    # #feature transformation
    #df['evaporation'] = np.log1p(df['evaporation']) #log transformation
    #df['rainfall'] = np.log1p(df['rainfall']) #log transformation

    #feature engineering
    df['year'] = pd.DatetimeIndex(df['date']).year
    df['month'] = pd.DatetimeIndex(df['date']).month
    df.drop(['mintemp', 'date', 'temp3pm', 'maxtemp'], axis = 1, inplace = True) #Dropping columns

    

    #Transformaing numerical features
    train_data, val_data,test_data, y_train, y_val, y_test = split_data(df)[0:6]
    cat_col = train_data.dtypes[train_data.dtypes == 'object'].index.tolist()
    num_col = train_data.dtypes[train_data.dtypes == 'float64'].index.tolist()
    
    scaler = StandardScaler()
    transformed_xtrain = scaler.fit_transform(train_data[num_col])
    transformed_xtest = scaler.transform(test_data[num_col])
    transformed_xval  = scaler.transform(val_data[num_col])

    train_df = pd.DataFrame(transformed_xtrain, columns = num_col)
    val_df = pd.DataFrame(transformed_xval, columns = num_col)
    test_df = pd.DataFrame(transformed_xtest, columns = num_col)

    train_df = pd.concat([train_df, train_data[cat_col]], axis = 1)
    val_df = pd.concat([val_df, val_data[cat_col]], axis = 1)
    test_df = pd.concat([test_df, test_data[cat_col]], axis = 1)

    train_df.drop(['pressure9am'], axis = 1, inplace = True)
    val_df.drop(['pressure9am'], axis = 1, inplace = True)
    test_df.drop(['pressure9am'], axis = 1, inplace = True)

    full_train_data = pd.concat([train_df, val_df])
    full_y_train = pd.concat([y_train, y_val])

    return scaler,full_train_data,full_y_train, test_df, y_test

#oversampling data
def data_augm(xtrain, ytrain):
        
    oversample = SMOTE()
    X_train, ytrain = oversample.fit_resample(xtrain, ytrain)
    return X_train, ytrain


print('Loading Data...')
print()
data = load_data('./Weather/weatherAUS.csv')  #loading data
print('Data loaded successfully')
print()

print('Preparing Data')
print()
scaler, train_data,y_train, test_data, y_test = data_prep(data) #data preparation
print('Data prepatation complete')
print()

print('Encoding Data...')
print()
dv, X_train,X_test = encoding(train_data, test_data) #data encoding
print('Data encoding complete')
print()

print('Augumenting Data...')
print()
X_train, y_train = data_augm(X_train, y_train) #data augmentation
print('Data augmentation complete')
print()

print('Training model...')
print()
precision, recall, f1score, model = xg_boost(X_train,y_train,X_test, y_test)
print('Model Evaluation')
print()

print('The model precision score is ', precision)
print('The model recall score is ', recall)
print('The model f1score is ', f1score)

with open(out, 'wb') as f:
    pickle.dump((model, scaler, dv), f)

