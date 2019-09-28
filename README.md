mlkit-learn
===========
![license](https://img.shields.io/github/license/ryanxjhan/mlkit-learn.svg)
![pyv](https://img.shields.io/pypi/pyversions/mklearn.svg)
![pypiv](https://img.shields.io/pypi/v/mklearn.svg?color=green)
![format](https://img.shields.io/pypi/format/mklearn.svg)

mlkit-learn is a lightweight machine learning library designed to be **interactive**, **easy-to-understand**, and **educational**. It implements all of the classic machine learning algorithms from regression to gradient boosting trees. **With only two lines of code, users can witness popular Kaggle datasets being preprocessed and predicted in action**.


## Contents
[Nearest Neighbor](#nearest-neighbor)

[Multivariate Linear Regression](#multivariate-linear-regression)

[Naive Bayes](#naive-bayes)

[Decision Tree](#decision-tree)

[Logistic Regression](#logistic-regression)


## Install
`pip install mklearn`


## Demo


### Nearest Neighbor
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
model = KNNClassifier(k)
model.fit(train_x, train_y)
size = test_x.shape[0]
predictions = []
for i in range(size):
    result = classifier.predict(test_x[i])
    predictions.append(result)
```

##### KNN Regressor

```python
model = KNNRegressor(k)
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
### Naive Bayes
### Decision Tree
### Logistic Regression



