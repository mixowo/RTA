import numpy as np
import sklearn
from sklearn import datasets
import pandas as pd
import pickle

from flask import Flask
from flask import request
from flask import jsonify

iris= datasets.load_iris()
df = pd.DataFrame(data = np.c_[iris["data"], iris["target"]], columns = iris["feature_names"]+["target"])

df.drop(df.index[df["target"] == 2], inplace = True)
y = df.loc[:, ["target"]].values
X = df.loc[:, ["petal length (cm)", "sepal length (cm)"]].values

class Perceptron():
    def __init__(self,eta=0.01, n=10):
        self.eta = eta
        self.n = n
    def fit(self, X, y):
        self.w_ = np.zeros(1+X.shape[1])
        self.errors_ = []
        for _ in range(self.n):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta*(target-self.predict(xi))
                self.w_[1:] += update*xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    def net_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]
    def predict(self, X):
        return np.where(self.net_input(X)>=1,1,0)
    
model = Perceptron()
model.fit(X, y)

with open("model.pkl", "wb") as model123:
    pickle.dump(model, model123)
    
    
app = Flask(__name__)

@app.route('/api/predict', methods=['GET'])
def get_prediction():
    sepal_length = float(request.args.get('sl'))
    petal_length = float(request.args.get('pl'))
    features = [sepal_length,
                petal_length]
    print(features)
    with open('model.pkl',"rb") as picklefile:
        model = pickle.load(picklefile)
    print(model)
    predicted_class = int(model.predict(features))
    return jsonify(features=features, predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')