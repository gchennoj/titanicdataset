import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


def load_data(train_path: str, test_path: str):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    data = pd.concat([train, test], axis=0, sort=False).reset_index(drop=True)
    return data, len(train)


def clean_data(data: pd.DataFrame):
    # Drop PassengerId and Cabin
    data.drop(columns=['PassengerId', 'Cabin'], inplace=True, errors='ignore')

    # Impute missing numeric with random sampling
    num_features = data.select_dtypes(include=['number']).columns.tolist()

    for feature in num_features:
        if data[feature].isnull().any():
            random_sample = data[feature].dropna().sample(data[feature].isnull().sum(), random_state=0)
            random_sample.index = data[data[feature].isnull()].index
            data.loc[data[feature].isnull(), feature] = random_sample

    # Impute missing categorical with mode
    cat_features = data.select_dtypes(include=['object']).columns.tolist()

    for feature in cat_features:
        mode_val = data[feature].mode()[0]
        data[feature] = data[feature].fillna(mode_val)

    return data


def feature_engineering(data: pd.DataFrame):
    # Create Family_members feature
    data['Family_members'] = data['SibSp'] + data['Parch']

    # Encode categorical columns
    cat_features = data.select_dtypes(include=['object']).columns.tolist()
    le = LabelEncoder()
    for feature in cat_features:
        data[feature] = le.fit_transform(data[feature])

    # Drop unnecessary columns
    data.drop(columns=['Name', 'Ticket'], axis=1, inplace=True, errors='ignore')
    return data


def plot_correlation(data: pd.DataFrame):
    numeric_data = data.select_dtypes(include=['number']).drop(columns=['Survived'], errors='ignore')
    correlation_matrix = numeric_data.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="Reds")
    plt.title("Correlation Heatmap (Numeric Columns Only)")
    plt.show()

    correlation_with_survival = data.select_dtypes(include=['number']).corr()['Survived']
    print("\nCorrelation with 'Survived':\n", correlation_with_survival.abs().sort_values(ascending=False)[1:])


def train_models(train_data: pd.DataFrame, target: pd.Series):
    models = {
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=10),
        "Random Forest": RandomForestClassifier()
    }

    for name, model in models.items():
        model.fit(train_data, target)
        preds = model.predict(train_data)
        acc = accuracy_score(target, preds)
        print(f"{name} Accuracy on Train Data: {acc:.4f}")


def main():
    # 1. Load Data
    data, train_len = load_data("titanic_dataset/train.csv", "titanic_dataset/test.csv")

    # 2. Clean Data
    data = clean_data(data)

    # 3. Feature Engineering
    data = feature_engineering(data)

    # 4. Correlation
    plot_correlation(data)

    # 5. Split Train/Test
    train_data = data.iloc[:train_len]
    test_data = data.iloc[train_len:].drop(columns=['Survived'], errors='ignore')
    target = train_data.pop('Survived')

    # 6. Train Models
    train_models(train_data, target)


if __name__ == "__main__":
    main()
