import json
import pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (accuracy_score, f1_score, 
                             precision_score, recall_score)
from utils_and_constants import PARAMETERS

def prep_data(X_train):

    categorical_col = X_train.dtypes[X_train.dtypes == 'object'].index.tolist()
    numerical_col = X_train.dtypes[X_train.dtypes == 'float64'].index.tolist()

    scaler = StandardScaler()
    X_train[numerical_col] = scaler.fit_transform(X_train[numerical_col])
    
    vectorizer = DictVectorizer(sparse = False)
    vectorizer.fit(X_train[categorical_col + numerical_col].to_dict(orient = 'records'))
    X_train = vectorizer.transform(X_train[categorical_col + numerical_col].to_dict(orient = 'records'))
    return X_train, scaler, vectorizer

def eval_metrics(y_true, prediction):

    f1 = f1_score(y_true, prediction)
    metrics = {"acc": accuracy_score(y_true, prediction), 
              "f1_score": f1, 
              "precision": precision_score(y_true, prediction), 
              "recall": recall_score(y_true, prediction)}
    return metrics

def train_model(X_train, y_train):

    with open(PARAMETERS, 'r') as json_file:
        parameters = json.load(json_file)
        parameters['max_depth'] = int(parameters['max_depth'])
    
    X_train, scaler, vectorizer = prep_data(X_train)
    train_data = xgb.DMatrix(X_train, label=y_train)
    
    booster = xgb.train(params=parameters,
                        dtrain=train_data,
                        num_boost_round=1000,
                        evals=[(train_data, 'validation')],
                        early_stopping_rounds=200
                        )

    prediction0 = booster.predict(train_data)
    prediction = (prediction0 >=0.5).astype('int')
    metrics = eval_metrics(y_train, prediction)
    
    return booster, scaler, vectorizer, metrics


def evaluate_model(model, scaler, vectorizer, X_test, 
                   y_test, float_precision=4):

    categorical_col = X_test.dtypes[X_test.dtypes == 'object'].index.tolist()
    numerical_col = X_test.dtypes[X_test.dtypes == 'float64'].index.tolist()

    X_test[numerical_col] = scaler.transform(X_test[numerical_col])
    X_test = vectorizer.transform((X_test[categorical_col + numerical_col].to_dict(orient = 'records')))
    X_test = xgb.DMatrix(X_test)

    y_proba = model.predict(X_test)
    prediction = (y_proba >=0.5).astype('int')
    metrics = eval_metrics(y_test, prediction)
    
    metrics = json.loads(json.dumps(metrics), 
                         parse_float=lambda x: round(float(x), float_precision))
    return metrics,y_proba

def save_model(model, scaler, vectorizer):

    model_name = '../models/model.pkl'
    with open(model_name, 'wb') as f:
        pickle.dump([model, scaler, vectorizer], f)
    print("Model saved successfully!")