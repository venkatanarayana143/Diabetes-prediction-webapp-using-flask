import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

data = pd.read_csv("diabetes.csv")
x = data.iloc[:,0:8].values

y = data.iloc[:, 8].values

train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.25, random_state=0)
#print(train_x)
sc = StandardScaler()
train_x = sc.fit_transform(train_x)

test_x = sc.fit_transform(test_x)

cls = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski", p=2)
cls.fit(train_x, train_y)
pickle.dump(cls,open("model.pkl", "wb"))

#model = pickle.load(open("model.pkl","rb"))
#print(model.predict([[1,89,66,23,94,28.1,0.67,21]]))
y_pred = cls.predict(test_x)
c_m = confusion_matrix(test_y, y_pred)
acc = accuracy_score(test_y,y_pred)

print(c_m)
print(acc)
