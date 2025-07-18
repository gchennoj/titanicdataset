import os
import sys
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import load as joblib_load
import pickle
from tensorflow.keras.models import load_model as keras_load_model
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


def load_test_data(test_path: str):
    test = pd.read_csv(test_path)
    test.drop(columns=['PassengerId', 'Cabin'], inplace=True, errors='ignore')

    num_features = test.select_dtypes(include=['number']).columns.tolist()
    for feature in num_features:
        if test[feature].isnull().any():
            random_sample = test[feature].dropna().sample(test[feature].isnull().sum(), random_state=0)
            random_sample.index = test[test[feature].isnull()].index
            test.loc[test[feature].isnull(), feature] = random_sample

    cat_features = test.select_dtypes(include=['object']).columns.tolist()
    for feature in cat_features:
        mode_val = test[feature].mode()[0]
        test[feature] = test[feature].fillna(mode_val)

    test['Family_members'] = test['SibSp'] + test['Parch']
    le = LabelEncoder()
    for feature in cat_features:
        test[feature] = le.fit_transform(test[feature])

    test.drop(columns=['Name', 'Ticket'], axis=1, inplace=True, errors='ignore')

    return test


def load_model_artifact(model_file):
    if model_file.endswith(".joblib"):
        return joblib_load(model_file)
    elif model_file.endswith(".pkl"):
        with open(model_file, 'rb') as f:
            return pickle.load(f)
    elif model_file.endswith(".h5"):
        return keras_load_model(model_file)
    else:
        raise ValueError(f"Unsupported model format: {model_file}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 test.py <model_artifact_path>")
        sys.exit(1)

    model_file = sys.argv[1]

    # Load the model artifact dynamically based on extension
    model = load_model_artifact(model_file)

    # Load test data
    test_data = load_test_data("titanic_dataset/test.csv")
    if 'Survived' not in test_data.columns:
        raise ValueError("Test dataset must contain 'Survived' column for evaluation.")

    target = test_data.pop('Survived')

    # Evaluate model
    predictions = model.predict(test_data)
    accuracy = accuracy_score(target, predictions)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Save metrics
    os.makedirs('/tmp/artifacts', exist_ok=True)
    metrics = {"accuracy": accuracy}
    with open('/tmp/artifacts/metrics.json', 'w') as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
