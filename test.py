import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

def create_model(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    print('Accuracy of our model:', accuracy_score(y_test, y_pred))
    print("Classification report:\n", classification_report(y_test, y_pred))

    return model, scaler

def get_clean_data(filepath=r'C:\Users\navee\Downloads\Breast-Cancer-Prediction--Logistic-Regresssion-main\data.csv'):
    if not os.path.exists(filepath):
        print(f"Error: The file at {filepath} was not found.")
        return None

    # Load and clean the data
    data = pd.read_csv(filepath)
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def main():
    data = get_clean_data()
    if data is not None:
        model, scaler = create_model(data)
        # Save model and scaler
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

if __name__ == '__main__':
    main()
