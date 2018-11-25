import pandas as pd
from sklearn import preprocessing
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn import ensemble
from sklearn.neighbors import KNeighborsClassifier
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression


le = preprocessing.LabelEncoder()

df_x = pd.read_csv('features_cat_6.csv')
X = df_x.iloc[:,1:519].values
y = df_x.iloc[:,520].values

pca = PCA(n_components = 'mle', svd_solver = 'full')
X = pca.fit_transform(X)

le.fit(y)
y = le.transform(y)
# y = np_utils.to_categorical(y)
print ("Label Encoded")

classifiers = [
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    ensemble.GradientBoostingClassifier(),
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial'),
    RandomForestClassifier(n_estimators=128),
    XGBClassifier(),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

(trainData, testData, trainLabels, testLabels) = train_test_split(X, y, test_size=0.25, random_state=42)

for clf in classifiers:
    print (clf.__class__.__name__)
    scores = cross_val_score(clf, X, y, cv=3)
    print (scores)