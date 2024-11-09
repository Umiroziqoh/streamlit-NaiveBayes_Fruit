import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Set the correct path to the data file
data_file =  'C:/Users/ASUS/OneDrive/문서/UTS Machinr Learning/fruit_data.xlsx'
model_file = 'NaiveBayes_Fruit.sav'

# Print the current working directory
print("Current working directory:", os.getcwd())

# Check if dataset exists
if not os.path.exists(data_file):
    print(f"Error: The dataset '{data_file}' does not exist.")
else:
    # Load the dataset
    df_fruit = pd.read_excel(data_file)

    # Encode the target column
    en = LabelEncoder()
    df_fruit['name'] = en.fit_transform(df_fruit['name'])

    # Split features and labels
    X = df_fruit.iloc[:, :-1].values
    y = df_fruit.iloc[:, -1].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Check if the model file exists
    if not os.path.exists(model_file):
        print(f"The model file '{model_file}' does not exist. Training a new model...")
        
        # Initialize the Naive Bayes classifier
        classifier = GaussianNB()
        
        # Train the model
        classifier.fit(X_train, y_train)
        
        # Save the model
        pickle.dump(classifier, open(model_file, 'wb'))
        print(f"Model trained and saved as '{model_file}'.")
    else:
        # Load the existing model
        classifier = pickle.load(open(model_file, 'rb'))
        print(f"Model loaded from '{model_file}'.")

    # Predict using the model
    y_pred = classifier.predict(X_test)

    # Print evaluation metrics
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
