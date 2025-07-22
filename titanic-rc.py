import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
import warnings
warnings.filterwarnings("ignore")


def load_data(train_path: str, test_path: str):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    data = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)
    return data, len(train)


def clean_data(data: pd.DataFrame):
    data.drop(columns=['PassengerId', 'Cabin'], inplace=True, errors='ignore')
    num_features = data.select_dtypes(include=['number']).columns.tolist()
    for feature in num_features:
        if data[feature].isnull().any():
            random_sample = data[feature].dropna().sample(data[feature].isnull().sum(), random_state=0)
            random_sample.index = data[data[feature].isnull()].index
            data.loc[data[feature].isnull(), feature] = random_sample
    cat_features = data.select_dtypes(include=['object']).columns.tolist()
    for feature in cat_features:
        mode_val = data[feature].mode()[0]
        data[feature] = data[feature].fillna(mode_val)
    return data


def feature_engineering(data: pd.DataFrame):
    data['Family_members'] = data['SibSp'] + data['Parch']
    cat_features = data.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    for feature in cat_features:
        data[feature] = le.fit_transform(data[feature])
    data.drop(columns=['Name', 'Ticket'], axis=1, inplace=True, errors='ignore')
    return data


def train_and_serialize_model(train_data: pd.DataFrame, target: pd.Series):
    model = RandomForestClassifier()
    model.fit(train_data, target)
    preds = model.predict(train_data)
    acc = accuracy_score(target, preds)
    print(f"Random Forest Accuracy on Train Data: {acc:.4f}")

    artifacts_dir = '/mnt/shared'
    os.makedirs(artifacts_dir, exist_ok=True)
    model_file = os.path.join(artifacts_dir, 'model.joblib')
    dump(model, model_file)

    metadata = {
        "serialization_format": "joblib",
        "model_file": "model.joblib"
    }
    with open(os.path.join(artifacts_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f)


def main():
    data, train_len = load_data("titanic_dataset/train.csv", "titanic_dataset/test.csv")
    data = clean_data(data)
    data = feature_engineering(data)
    train_data = data.iloc[:train_len]
    target = train_data.pop('Survived')

    train_and_serialize_model(train_data, target)


if __name__ == "__main__":
    main()
