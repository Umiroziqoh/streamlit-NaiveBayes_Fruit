import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Load the dataset from a predefined path
df_fruit = pd.read_excel('fruit_data.xlsx')

# Check the first few rows of the dataset
print("Data Preview:")
print(df_fruit.head())

# Check the structure of the dataset
print("Dataset Info:")
print(df_fruit.info())

# Check if the dataset is empty
if df_fruit.empty:
    print("Dataset is empty!")
else:
    # Encode the fruit names into numeric labels
    en = LabelEncoder()
    df_fruit['name'] = en.fit_transform(df_fruit['name'])

    # Prepare the feature matrix (X) and target vector (y)
    x = df_fruit.iloc[:, :-1].values
    y = df_fruit.iloc[:, -1].values

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

    print(f"x_train: {len(x_train)}")
    print(f"x_test: {len(x_test)}")
    print(f"y_train: {len(y_train)}")
    print(f"y_test: {len(y_test)}")

    # Standardize the features
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # Initialize and train the Naive Bayes classifier
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)

    # Make predictions
    y_pred = classifier.predict(x_test)

    # Predict probability of each class
    class_probabilities = classifier.predict_proba(x_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Accuracy Score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Create a DataFrame to store the actual and predicted values
    ydata = pd.DataFrame()
    ydata['y_test'] = pd.DataFrame(y_test)
    ydata['y_pred'] = pd.DataFrame(y_pred)
    print("Actual vs Predicted Values:")
    print(ydata)

    # Save the results to an Excel file
    ydata.to_excel('data_fruit_actualpred.xlsx', index=False)
    print("Prediction results saved to 'data_fruit_actualpred.xlsx'.")

    # Save the trained model using pickle
    filename = 'NaiveBayes_Fruit.sav'
    pickle.dump(classifier, open(filename, 'wb'))
    print(f"Model saved as {filename}")

    # Example: Load and test the saved model
    loaded_model = pickle.load(open(filename, 'rb'))
    test_sample = np.array([x_test[0]])  # Take a sample from the test set
    test_sample = sc.transform(test_sample)  # Standardize the sample
    prediction = loaded_model.predict(test_sample)
    predicted_fruit = en.inverse_transform(prediction)
    print(f"Prediction for sample: {predicted_fruit[0]}")
