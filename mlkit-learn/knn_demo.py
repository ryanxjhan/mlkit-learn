import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from knn import KNNClassifier

def load_demo(row, data_dir="datasets/"):
    data = pd.read_csv(data_dir + "digit.csv")
    x = data.values[0:row, 1:]
    y = data.values[0:row, 0]
    return x, y

def split_demo(x, y):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)
    return train_x, test_x, train_y, test_y

def predict_demo(train_x, test_x, train_y, test_y):
    classifier = KNNClassifier(5)
    print("{} loaded.".format(str(classifier)))
    classifier.fit(train_x, train_y)
    size = test_x.shape[0]
    predictions = []
    for i in range(size):
        result = classifier.predict(test_x[i])
        predictions.append(result)
    print("Prediction completes.")
    print("Validation score: {}".format(accuracy_score(test_y, predictions)))

if __name__ == "__main__": 
    print("See datasets source on: https://www.kaggle.com/c/digit-recognizer/data")
    print("Only the first 1000 rows are used for faster prediction.")    
    x, y = load_demo(1000)
    print("Demo data loaded.")
    train_x, test_x, train_y, test_y = split_demo(x, y)
    print("Data splitted for validation.")
    predict_demo(train_x, test_x, train_y, test_y)
    print("To learn more about K-Nearest Neighbor Algorithm on https://github.com/ryanxjhan/mlkit-learn")
