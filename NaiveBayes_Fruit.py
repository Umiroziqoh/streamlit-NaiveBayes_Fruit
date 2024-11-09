import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pickle

# Title of the app
st.title('Fruit Classification Web Application')

# Upload the fruit data file
uploaded_file = st.file_uploader("Upload Fruit Data Excel File", type=["xlsx"])

if uploaded_file is not None:
    df_fruit = pd.read_excel(uploaded_file)
    st.write("Data Preview:")
    st.write(df_fruit.head())
    
    # Check if data is loaded successfully
    if df_fruit.empty:
        st.error("The uploaded file is empty.")
    else:
        # Preprocessing the data
        en = LabelEncoder()
        df_fruit['name'] = en.fit_transform(df_fruit['name'])

        # Split the features and labels
        x = df_fruit.iloc[:, :-1].values
        y = df_fruit.iloc[:, -1].values

        # Split the dataset into training and testing
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
        
        # Standardize the data
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        # Train the model
        classifier = GaussianNB()
        classifier.fit(x_train, y_train)
        
        # Predict the results
        y_pred = classifier.predict(x_test)
        
        # Show metrics
        st.subheader('Model Accuracy')
        accuracy = accuracy_score(y_test, y_pred) * 100
        st.write(f"Accuracy: {accuracy:.2f}%")
        
        # Show confusion matrix
        st.subheader('Confusion Matrix')
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)
        
        # Show classification report
        st.subheader('Classification Report')
        report = classification_report(y_test, y_pred)
        st.text(report)

        # Allow user to input new data for prediction
        st.subheader('Fruit Prediction')
        input_features = []
        
        # Dynamically create input fields based on the features
        for col in df_fruit.columns[:-1]:  # Exclude the 'name' column (target)
            input_value = st.number_input(f"Enter value for {col}", value=0)
            input_features.append(input_value)
        
        if st.button('Predict Fruit Type'):
            input_features = np.array(input_features).reshape(1, -1)
            input_features = sc.transform(input_features)  # Standardize input data
            prediction = classifier.predict(input_features)
            predicted_class = en.inverse_transform(prediction)
            st.write(f"The predicted fruit is: {predicted_class[0]}")
        
        # Save the model
        if st.button('Save Model'):
            filename = 'NaiveBayes_Fruit.sav'
            pickle.dump(classifier, open(filename, 'wb'))
            st.write(f"Model saved as {filename}")

        # Provide an option to download the predicted data
        if st.button('Download Predictions'):
            ydata = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
            output_file = 'data_fruit_actualpred.xlsx'
            ydata.to_excel(output_file, index=False)
            st.write(f"Download the prediction data: [Click here to download]({output_file})")
