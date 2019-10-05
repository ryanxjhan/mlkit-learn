import numpy as np

class NBClassifier:

    def __str__(self):
        return "Naive Bayes Classifier"

    def info(self):
        print("Naive Bayes Classifier:")
        print("Pro: works with a small amount of data.")
        print("Cons: sensitive to how the input data is prepared.")
        print("- Machine Learning in Action")

    def fit(self, x, y):

        row = len(x)
        col = len(x[0])

        # count every individual class
        count = {}
        total_count = {}
        vec_count = {}
        for i in np.unique(y):
            count[i] = 0
            total_count[i] = 0
            vec_count[i] = np.ones(col)
        
        for i in range(row):
            vec_count[i] += x[i]
            total_count += sum(x[i])
            count[i] += 1

        prob = []

        for i in count:
            prob.append(np.log(vec_count[i] / (total_count[i] + col)))
            
        return prob, 
        
    def predict(self, ):
        

