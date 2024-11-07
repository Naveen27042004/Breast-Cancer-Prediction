import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def load_and_clean_data(filepath="data.csv"):
    """
    Load and clean the dataset from the specified CSV file.
    
    Parameters:
    - filepath (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Cleaned data with features and target.
    """
    try:
        # Load the data
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file at {filepath} was not found.")
        return None

    # Drop unnecessary columns
    data = data.drop(['Unnamed: 32', 'id'], axis=1)

    # Encode diagnosis: M -> 1, B -> 0
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    
    return data


def create_and_train_model(data):
    """
    Creates, trains, and evaluates a logistic regression model.
    
    Parameters:
    - data (pd.DataFrame): The cleaned dataset with features and target.

    Returns:
    - model (LogisticRegression): Trained logistic regression model.
    - scaler (StandardScaler): Scaler fitted on the feature data.
    """
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    print('Model Accuracy:', accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model, scaler


def main():
    # Load and clean the data
    data = load_and_clean_data()

    # Ensure data was loaded successfully
    if data is not None:
        # Create, train, and evaluate the model
        model, scaler = create_and_train_model(data)


if __name__ == '__main__':
    main()
