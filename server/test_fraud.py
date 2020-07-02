from joblib import dump, load
import pandas as pd
from fraud_classifier import FraudClassifier
from pandas.io.json import json_normalize

MODEL_NAME = '2020-06-14'
TRAIN_DATA_DIR = 'data/SupervisedChallenge.json'
MODEL_PERSISTENT_DIR = 'model_persistent'


def train():
    df = pd.read_json(TRAIN_DATA_DIR, orient='records', lines=True)
    model = FraudClassifier()
    model.fit(df)
    dump(model, f'{MODEL_PERSISTENT_DIR}/{MODEL_NAME}.joblib')


def prediction(X):
    X = json_normalize(X)
    if X['refused_by_bank'].values[0] == 1:
        pred_y = 0
    else:
        model = load(f'{MODEL_PERSISTENT_DIR}/{MODEL_NAME}.joblib')
        pred_y = model.predict(X)
    return pred_y


if __name__ == '__main__':
    train()
