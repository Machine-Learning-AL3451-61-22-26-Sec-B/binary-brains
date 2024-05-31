import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data(file):
    data = pd.read_csv(file)
    return data

def preprocess_data(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Convert categorical variables to numerical
    label_encoders = {}
    for column in X.columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    le_PlayTennis = LabelEncoder()
    y = le_PlayTennis.fit_transform(y)

    return X, y, label_encoders

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    return classifier, X_test, y_test

def evaluate_model(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    return accuracy

def main():
    st.title('Tennis Prediction App')

    # File uploader for data
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        data = load_data(uploaded_file)

        # Display first 5 rows of data
        st.write("The first 5 values of data are:")
        st.write(data.head())

        # Preprocess data
        X, y, label_encoders = preprocess_data(data)

        # Train model
        classifier, X_test, y_test = train_model(X, y)

        # Evaluate model
        accuracy = evaluate_model(classifier, X_test, y_test)
        st.write("Accuracy of the model:", accuracy)

if __name__ == "__main__":
    main()
