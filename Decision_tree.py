import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OrdinalEncoder

cell_concentration = pd.read_csv('./data/raman_data.csv')

labels = np.array(cell_concentration['condition'])
frequency_list = np.array(cell_concentration.columns)[1:]
features = np.array(cell_concentration.iloc[:,1:])

temp_x = np.array(cell_concentration.columns)[1:].astype(float)
frequency_list = temp_x[np.where(np.logical_and(temp_x>=800, temp_x<=1800))]

features = np.array(cell_concentration.loc[:,frequency_list.astype(str)])

encoder = OrdinalEncoder()
encoder.fit(np.reshape(labels,(-1,1)))
labels = encoder.transform(np.reshape(labels,(-1,1)))

# Normalize
features = (features - np.min(features,0))/(np.max(features,0)-np.min(features,0))

# Split Data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.001, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

clf.score(X_test,y_test)

clf = DecisionTreeClassifier()
cross_val_score(clf, X_train, y_train, cv=10)