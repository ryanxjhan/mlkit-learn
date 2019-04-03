import numpy as np

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
     