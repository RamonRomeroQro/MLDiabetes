
# dependencies
from sklearn.preprocessing import StandardScaler  # Standardisation
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import copy
from sklearn.externals.joblib import dump, load
import warnings
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
from subprocess import check_output

print("Reading data and endoding")

a = pd.read_csv('./data/2016.csv', header=0)
b = pd.read_csv('./data/2016_2.csv', header=0)
c = pd.read_csv('./data/2017.csv', header=0)
d = pd.read_csv('./data/2018.csv', header=0)
# key atributes
important = ["IDE_EDA_ANO", "IDE_SEX", "DIAB_PAD_MAD", "DIAB_HER", "DIAB_HIJ",
             "DIAB_OTROS", "CVE_ACT_FIS", "CVE_TAB", "CVE_COMB_TUBER",
             "CVE_COMB_CANCER", "CVE_COMB_OBESIDAD", "CVE_COMB_HIPER",
             "CVE_COMB_VIH_SIDA", "CVE_COMB_DEPRE", "CVE_COMB_DISLI", "CVE_COMB_CARDIO",
             "CVE_COMB_HEPA", "CVE_NUT", "CVE_OFT",
             "CVE_PIES", "CVE_DIAB", "CVE_TIPO_DISC_MOTO", "CVE_TIPO_DISC_VISU",
             "PESO", "ESTATURA"]

# label
expected = "CVE_DIAB"

# vertical join

result = pd.concat([a, b, c, d])

# hot encoding

result = result.replace("Masculino", 1)
result = result.replace("Femenino", 0)
result = result.replace("Si", 1)
result = result.replace("No", 0)
result["CVE_TAB"] = result.apply(lambda row: 0 if "Nunca" in str(row["CVE_TAB"]) else 1,
                                 axis=1)

result.fillna(0)
result["CVE_NUT"] = result.apply(lambda row: 0 if "Nunca" in str(row["CVE_NUT"]) else 1,
                                 axis=1)

result["CVE_OFT"] = result.apply(lambda row: 0 if "Nunca" in str(row["CVE_OFT"]) else 1,
                                 axis=1)

result["CVE_PIES"] = result.apply(lambda row: 0 if "Nunca" in str(row["CVE_PIES"]) else 1,
                                  axis=1)
result = result[important]


diab = result

important.remove(expected)


warnings.filterwarnings('ignore')


# train split 1
outcome = diab[expected]
data = diab[important]
train, test = train_test_split(
    diab, test_size=0.20, random_state=0, stratify=diab[expected])  # stratify the outcome
train_X = train[important]
test_X = test[important]
train_Y = train[expected]
test_Y = test[expected]

print("Training models")


# simple models generation

abc = []
classifiers = ['Linear Svm', 'Radial Svm',
               'Logistic Regression', 'KNN', 'Decision Tree', 'One Class SVM']
models = [svm.SVC(kernel='linear'), svm.SVC(kernel='rbf'), LogisticRegression(
), KNeighborsClassifier(n_neighbors=3), DecisionTreeClassifier(), svm.OneClassSVM(kernel="rbf")]
simple_models = {}

for i in range(len(models)):
    model = models[i]
    model.fit(train_X, train_Y)
    simple_models[classifiers[i]] = model
    prediction = model.predict(test_X)
    abc.append(metrics.accuracy_score(prediction, test_Y))
models_dataframe = pd.DataFrame(abc, index=classifiers)
models_dataframe.columns = ['Accuracy']

print("Saving simple models")

# saving simple models


for name, model in simple_models.items():
    filename = f"./outputs/simple_{name}.save"
    dump(model, filename)


# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))

# Feature Extraction/ Selection:


model = RandomForestClassifier(n_estimators=100, random_state=0)
X = diab[important]
Y = diab[expected]
model.fit(X, Y)

serie = pd.Series(model.feature_importances_,
                  index=X.columns).sort_values(ascending=False)
d = serie.to_dict()


sorted_d = sorted(d.items(), key=lambda kv: kv[1])[::-1]


####  three more important by class  ####

argc = 3

print(f"Feature Extraction ({argc})")

weighted = []
for i in range(3):
    weighted.append(sorted_d[i][0])

cut = copy.deepcopy(weighted)
cut.append(expected)


diab2 = diab[cut]


features = diab2[weighted]
scaler = StandardScaler()
features_standard = scaler.fit_transform(features)  # Gaussian Standardisation
##########

print("StandardScaler")


scaler_filename = "./outputs/scaler.save"
joblib.dump(scaler, scaler_filename)

# And now to load...

# scaler = joblib.load(scaler_filename)

# scaled_instances = scaler.transform(raw_instances)


###########
x = pd.DataFrame(features_standard, columns=[weighted])


x[expected] = diab2[expected].values


outcome = x[expected]
train1, test1 = train_test_split(
    x, test_size=0.20, random_state=0, stratify=x[expected])
train_X1 = train1[weighted]
test_X1 = test1[weighted]
train_Y1 = train1[expected]
test_Y1 = test1[expected]

print("training selected models")


select_models = {}
abc = []
classifiers = ['Linear Svm', 'Radial Svm',
               'Logistic Regression', 'KNN', 'Decision Tree', 'One Class SVM']
models = [svm.SVC(kernel='linear'), svm.SVC(kernel='rbf'), LogisticRegression(
), KNeighborsClassifier(n_neighbors=3), DecisionTreeClassifier(), svm.OneClassSVM(kernel="rbf")]


for i in range(len(models)):
    model = models[i]
    model.fit(train_X1, train_Y1)
    select_models[classifiers[i]] = model
    prediction = model.predict(test_X1)
    abc.append(metrics.accuracy_score(prediction, test_Y1))
new_models_dataframe = pd.DataFrame(abc, index=classifiers)
new_models_dataframe.columns = ['New Accuracy']


# In[104]:


#


# saving simple models

print("saving selected models")


for name, model in select_models.items():
    filename = f"./outputs/select_{name}.save"
    dump(model, filename)


# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
