import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class MLR:

    def __str__(self):
        return "Multivariate Linear Regression"
    
    def fit(self, x, y):
        self.x = x
        self.y = y
        
    def coef(self):
        xtx = np.dot(self.x.transpose(), self.x)
        xtx_inv = np.linalg.inv(xtx) 
        xty = np.dot(self.x.transpose(), self.y) 
        return np.dot(xtx_inv, xty.transpose())
    
    def predict(self, v):
        result = np.dot(v, self.coef())
        return result     
    
def demo(row=5000):
    try:
        data = pd.read_csv("kc_house_data.csv")
        print("Data loaded successfully.")
    except:
        print("Can not find kc_house_data.csv in the directory specified.")
        print("To download, please visit https://www.kaggle.com/harlfoxem/housesalesprediction/download#kc_house_data.csv.")
    x = data[['bedrooms', 'bathrooms', 'sqft_living', 'floors']]
    y = data['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    print("Data splitted for validation.")
    x_train = np.asmatrix(x_train)
    x_train = np.append(np.asmatrix(np.tile(1, x_train.shape[0])).transpose(), x_train, axis=1)
    x_test = np.append(np.asmatrix(np.tile(1, x_test.shape[0])).transpose(), x_test, axis=1)
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    model = MLR()
    model.fit(x_train, y_train)
    coef = model.coef()
    print("The coeffients are {!s}".format(coef))
    predictions = model.predict(x_test)
    print("The first five predictions are {!s}".format(predictions[:5]))
    print("The actual values are {!s}".format(y_test[:5]))
    abrd = (abs(np.array(predictions).transpose() - y_test) / y_test).sum() / len(y_test)
    print("The average absolute relative deviation is: {}".format(abrd))
    print("To learn more about Multivariate Linear Regression: https://en.wikipedia.org/wiki/Linear_regression")