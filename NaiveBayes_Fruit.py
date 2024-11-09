import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Load data
df_fruit = pd.read_excel('fruit_data.xlsx')

# Data Exploration
print(df_fruit.head())
print(df_fruit.info())
print(df_fruit.describe())  # Summary statistics

# Visualize data (example: histogram of fruit diameters)
plt.hist(df_fruit['diameter'], bins=10)
plt.xlabel('Diameter')
plt.ylabel('Frequency')
plt.title('Histogram of Fruit Diameters')
plt.show()

# Data Preprocessing
en = LabelEncoder()
df_fruit['name'] = en.fit_transform(df_fruit['name'])

# Split data
x = df_fruit.iloc[:, :-1].values
y = df_fruit.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

# Scale features
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Train Naive Bayes model
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Make predictions
y_pred = classifier.predict(x_test)

# Evaluate model
cm = confusion_matrix(y_test, y_pred)
print(cm)
akurasi = classification_report(y_test, y_pred)
print(akurasi)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Cross-validation
scores = cross_val_score(classifier, x, y, cv=5)  # 5-fold cross-validation
print("Cross-validation scores:", scores)
print("Average cross-validation accuracy: {:.2f}%".format(scores.mean() * 100))

# Compare with other models
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred_logreg = logreg.predict(x_test)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print("Logistic Regression Accuracy: {:.2f}%".format(accuracy_logreg * 100))

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Accuracy: {:.2f}%".format(accuracy_dt * 100))

# Save model
import pickle
filename = 'NaiveBayes_Fruit.sav'
pickle.dump(classifier, open(filename, 'wb'))
