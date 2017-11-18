from pandas import set_option
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

seed = 7
np.random.seed(seed)

file = "creditcard.csv"
data = pd.read_csv(file)

data = data.dropna()
data = data.drop_duplicates(subset=None, keep='first', inplace=False)
count = data.count()
types = data.dtypes
headers = data.head()

healthy_indices = data[data.Class == 0].index
fraud_indices = data[data.Class == 1].index
random_indices = np.random.choice(healthy_indices, 1000, replace=False)
new_sample = data.loc[random_indices]
new_f_sample = data.loc[fraud_indices]
data = data[0:0]

data = data.append(new_sample, ignore_index=True)
data = data.append(new_f_sample, ignore_index=True)

new_array = data.values
X = new_array[:, 0:30]
Y = new_array[:, 30]

sm = SMOTE(random_state=seed, ratio=1.0)
X, Y = sm.fit_sample(X, Y)
print(len(X))

set_option('display.width', 400)
set_option('precision', 4)
description = data.describe()
data_correlation = data.corr(method='pearson')

mod = ExtraTreesClassifier()
mod.fit(X, Y)

data = data.drop(data.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 16, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]],
                 axis=1)

oversampling = data.values
oX = oversampling[:, 0:6]
oY = oversampling[:, 6]

sm = SMOTE(random_state=seed, ratio=1.0)
X, Y = sm.fit_sample(oX, oY)

scaler = Normalizer().fit(X)
X = scaler.transform(X)

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
                                                                test_size=0.2, random_state=seed)

results = []
names = []

models = [('LR', LogisticRegression()), ('LDA', LinearDiscriminantAnalysis()), ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()), ('SVM', SVC())]

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='roc_auc')
    results.append(cv_results)
    names.append(name)
    final_results = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(final_results)

knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)

print(" ")
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
