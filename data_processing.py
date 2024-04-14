import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def preprocess_data(titanic_data_url):
    titanic_df = pd.read_csv(titanic_data_url)
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    target = 'Survived'
    titanic_df = titanic_df.dropna(subset=features + [target])
    titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})
    titanic_df['Embarked'] = titanic_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    X_train, X_test, y_train, y_test = train_test_split(titanic_df[features], titanic_df[target], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model