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
To use Nearest Neighbor's demo:
1. [download test.csv](https://www.kaggle.com/c/digit-recognizer/data) and put in your current directory.
2. run the following code.
```python
import mklearn.knn as knn
knn = knn.KNNDemo(1000, 5) # here we only use the first 1000 rows of the data and set k = 5.
knn.run() 
```
### Multivariate Linear Regression
### Naive Bayes
### Decision Tree
### Logistic Regression



