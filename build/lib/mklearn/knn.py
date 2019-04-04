import sys

try:
    import numpy as np
except ImportError as error:
    print(error)
    sys.exit(1)

try:
    import pandas as pd
except ImportError as error:
    print(error)
    sys.exit(1)

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except ImportError as error:
    print(error)
    sys.exit(1)


class KNNBase:
    
    def __init__(self, k):
        self.k = k
    
    def fit(self, x, y):
        self.x = x
        self.y = y
    
class KNNClassifier(KNNBase):
    
    def __str__(self):
        return "KNNClassifier({!s})".format(self.k)
    
    def predict(self, data):
        size = self.x.shape[0]
        diff_matrix = np.tile(data, (size, 1)) - self.x
        sq_matrix = diff_matrix ** 2
        sum_matrix = sq_matrix.sum(axis=1)
        distances = sum_matrix ** 0.5
        sorted_indicies = distances.argsort()
        
        class_count = {}
        for i in range(self.k):
            vote = self.y[sorted_indicies[i]]
            class_count[vote] = class_count.get(vote, 0) + 1
        sorted_class_count = sorted(class_count.items(), reverse=True)
        
        return sorted_class_count[0][0]
    
class KNNRegressor(KNNBase):
    
    def __str__(self):
        return "KNNRegressor({!s})".format(self.k)

    def predict(self, data):
        size = self.x.shape[0]
        diff_matrix = np.tile(data, (size, 1)) - self.x
        sq_matrix = diff_matrix ** 2
        sum_matrix = sq_matrix.sum(axis=1)
        distances = sum_matrix ** 0.5
        sorted_indicies = distances.argsort()
        
        total_sum = 0
        for i in range(self.k):
            total_sum += self.y[sorted_indicies[i]]
            
        return total_sum / self.k

class KNNDemo:

    def __init__(self, row, k):
        self.row = row
        self.k = k

    def load_data(self, data_dir=""):
        data = pd.read_csv(data_dir + "train.csv")
        x = data.values[0:self.row, 1:]
        y = data.values[0:self.row, 0]
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)
        return train_x, test_x, train_y, test_y
    
    def predict(self, train_x, test_x, train_y, test_y):
        classifier = KNNClassifier(self.k)
        print("{} loaded.".format(str(classifier)))
        classifier.fit(train_x, train_y)
        size = test_x.shape[0]
        predictions = []
        for i in range(size):
            result = classifier.predict(test_x[i])
            predictions.append(result)
        print("Prediction completes.")
        print("Validation score: {}".format(accuracy_score(test_y, predictions)))

    def run(self):   
        try:
            train_x, test_x, train_y, test_y = self.load_data()
        except:
            print("Can not find train.csv in your current directory.")
            print("To download, please visit https://www.kaggle.com/c/digit-recognizer/data.")
            sys.exit(1)
        print("Demo data loaded.")
        print("Data splitted for validation.")
        self.predict(train_x, test_x, train_y, test_y)
        print("To learn more about K-Nearest Neighbor Algorithm: https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm")