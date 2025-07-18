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
    model = load_model_artifact(model_file)

    test_data = load_test_data("titanic_dataset/test.csv")
    artifacts_dir = '/tmp/artifacts'
    os.makedirs(artifacts_dir, exist_ok=True)

    if 'Survived' in test_data.columns:
        target = test_data.pop('Survived')
        predictions = model.predict(test_data)
        accuracy = accuracy_score(target, predictions)
        print(f"Test Accuracy: {accuracy:.4f}")

        metrics = {"accuracy": accuracy}
        with open(os.path.join(artifacts_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)
        print(f"Metrics saved to {artifacts_dir}/metrics.json")
    else:
        predictions = model.predict(test_data)
        test_data['Predicted_Survived'] = predictions
        output_file = os.path.join(artifacts_dir, 'predictions.csv')
        test_data.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    main()
