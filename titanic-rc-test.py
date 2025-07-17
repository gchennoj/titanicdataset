import os
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import load
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


def load_test_data(test_path: str):
    test = pd.read_csv(test_path)
    test.drop(columns=['PassengerId', 'Cabin'], inplace=True, errors='ignore')

    # Fill missing numerics with random sampling from existing values
    num_features = test.select_dtypes(include=['number']).columns.tolist()
    for feature in num_features:
        if test[feature].isnull().any():
            random_sample = test[feature].dropna().sample(test[feature].isnull().sum(), random_state=0)
            random_sample.index = test[test[feature].isnull()].index
            test.loc[test[feature].isnull(), feature] = random_sample

    # Fill missing categoricals with mode
    cat_features = test.select_dtypes(include=['object']).columns.tolist()
    for feature in cat_features:
        mode_val = test[feature].mode()[0]
        test[feature] = test[feature].fillna(mode_val)

    # Feature engineering
    test['Family_members'] = test['SibSp'] + test['Parch']
    cat_features = test.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    for feature in cat_features:
        test[feature] = le.fit_transform(test[feature])

    # Drop unnecessary columns
    test.drop(columns=['Name', 'Ticket'], axis=1, inplace=True, errors='ignore')

    return test


def main():
    # Load test data
    test_data = load_test_data("titanic_dataset/test.csv")
    if 'Survived' not in test_data.columns:
        raise ValueError("Test dataset must contain 'Survived' column for evaluation.")

    target = test_data.pop('Survived')

    # Load the trained model artifact
    model = load('/tmp/artifacts/model.joblib')

    # Evaluate model on test data
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
