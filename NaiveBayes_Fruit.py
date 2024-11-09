import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pickle

app = Flask(__name__)

# Load model
model_filename = 'NaiveBayes_Fruit.sav'
classifier = pickle.load(open(model_filename, 'rb'))
scaler = StandardScaler()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        df_fruit = pd.read_excel(file)

        # Preprocessing
        en = LabelEncoder()
        df_fruit['name'] = en.fit_transform(df_fruit['name'])

        x = df_fruit.iloc[:, :-1].values
        y = df_fruit.iloc[:, -1].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # Prediction
        y_pred = classifier.predict(x_test)

        # Prepare results
        results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        results_excel = 'data_fruit_actualpred.xlsx'
        results.to_excel(results_excel, index=False)

        return f'Predictions saved to {results_excel}'

if __name__ == '__main__':
    app.run(debug=True)
