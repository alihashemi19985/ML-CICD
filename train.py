import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris  
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import  pickle

data = load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=20, random_state=42)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

with open("model.pkl", "wb") as file:
    pickle.dump(clf, file)

    