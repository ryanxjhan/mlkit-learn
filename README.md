mlkit-learn
===========
![license](https://img.shields.io/github/license/ryanxjhan/mlkit-learn.svg)
![pyv](https://img.shields.io/pypi/pyversions/mklearn.svg)
![pypiv](https://img.shields.io/pypi/v/mklearn.svg?color=green)
![format](https://img.shields.io/pypi/format/mklearn.svg)

mlkit-learn is a lightweight machine learning package designed to be **interactive**, **easy-to-understand**, and **educational**. It implements five of the classic machine learning algorithms from regression to gradient boosting trees. **With only two lines of code, users can witness popular Kaggle datasets being preprocessed and predicted in action**.


## Contents
[K-Nearest Neighbors](#k-nearest-neighbors)

[Multivariate Linear Regression](#multivariate-linear-regression)

[Naive Bayes](#naive-bayes)

[Decision Tree](#decision-tree)

[Logistic Regression](#logistic-regression)

[K-Means](#k-means)

[Support Vector Machine](#support-vector-machine)


## Install
`pip install mklearn`


## Demo


### K-Nearest Neighbor
#### Demo
1. download [train.csv](https://www.kaggle.com/c/digit-recognizer/data) and put in the directory of your choosing.
2. run the following code.
```python
from mklearn import knn
knn.demo(5, row=1000) # k [the number of nearest neighbour], dir [default: current directory], row [default: first 5000 rows]
```
#### Use

##### KNN Classifier

```python
from mklearn import knn
model = knn.KNNClassifier(5)
model.fit(train_x, train_y)
size = test_x.shape[0]
predictions = []
for i in range(size):
    result = classifier.predict(test_x[i])
    predictions.append(result)
```

##### KNN Regressor

```python
from mklearn import knn
model = knn.KNNRegressor(5)
model.fit(train_x, train_y)
size = test_x.shape[0]
predictions = []
for i in range(size):
    result = classifier.predict(test_x[i])
    predictions.append(result)
```

### Multivariate Linear Regression
#### Demo
1. download [kc_house_data.csv](https://www.kaggle.com/harlfoxem/housesalesprediction/download#kc_house_data.csv) and put in the directory of your choosing.
2. run the following code.
```python
from mklearn import mlr
mlr.demo(row=1000) # row [default: first 5000 rows]
```
#### Use

```python
from mklearn import mlr
x_train = np.asmatrix(x_train)
x_train = np.append(np.asmatrix(np.tile(1, x_train.shape[0])).transpose(), x_train, axis=1)
x_test = np.append(np.asmatrix(np.tile(1, x_test.shape[0])).transpose(), x_test, axis=1)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
model = mlr.MLR()
model.fit(x_train, y_train)
coef = model.coef()
predictions = model.predict(x_test)
```
### Naive Bayes
### Decision Tree
### Logistic Regression
### K-Means
### Support Vector Machine



