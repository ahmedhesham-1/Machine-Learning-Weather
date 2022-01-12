import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

dataset = pd.read_csv('/content/drive/MyDrive/weather_features.csv') 
dataset = dataset.dropna() 

dataset['weather_main'].unique()

dataset['city_name'].unique()

le = LabelEncoder()
dataset["dt_iso"] = le.fit_transform(dataset["dt_iso"]) 
dataset["city_name"] = le.fit_transform(dataset["city_name"])
dataset["weather_description"] = le.fit_transform(dataset["weather_description"])
dataset["weather_icon"] = le.fit_transform(dataset["weather_icon"])
dataset


x = dataset.drop(['weather_main'], axis=1) 
y = dataset['weather_main']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.25)
classifier = KNeighborsClassifier(n_neighbors=3) 
classifier.fit(xtrain, ytrain)

y_pred = classifier.predict(xtest)


print ("Training Accuracy: {}".format(classifier.score(xtrain, ytrain)))
predicted = classifier.predict(xtest)

print ("Testing Accuracy: {}".format(classifier.score(xtest,ytest)))

cm = confusion_matrix(ytest, y_pred)
print ("Confusion Matrix : \n", cm)
print (classification_report(ytest, y_pred))



