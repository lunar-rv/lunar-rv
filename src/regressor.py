print("Loading weighted regressor...")
import numpy as np
from sklearn.linear_model import LinearRegression

class LargeWeightsRegressor:
    def __init__(self, sensor_index=0):
        self.coef_ = None
        self.sensors_used = None
        self.indices_used = None
        self.sensor_index = sensor_index
        self.filter_model = LinearRegression(positive=True, fit_intercept=False) #Lasso(positive=True, alpha=1e-8)
        self.fit_model = LinearRegression(positive=True, fit_intercept=False)

    def choose_top_weight_indices(self, weights):
        sorted_indices = np.flip(weights.argsort())
        total_used = 0
        threshold = 0.8
        num_connections = 10
        indices_used = []
        for index in sorted_indices[:num_connections]:
            total_used += weights[index]
            indices_used.append(index)
            if total_used > threshold:
                break
        return np.array(indices_used)

    def fit(self, X, y):
        self.filter_model.fit(X, y)
        coef = self.filter_model.coef_
        sums = X.mean(axis=0)
        weighted_coef = coef / sums
        normalized_coef = weighted_coef / weighted_coef.sum()
        self.indices_used = self.choose_top_weight_indices(normalized_coef)
        self.sensors_used = np.array([i if i < self.sensor_index else i + 1 for i in self.indices_used])
        filtered_X = X[:, self.indices_used]
        self.fit_model.fit(filtered_X, y)
        self.coef_ = self.fit_model.coef_

    def predict(self, X):
        filtered_X = X[:, self.indices_used]
        return self.fit_model.predict(filtered_X)
    
    def set_sensor_index(self, sensor_index: int):
        self.sensor_index = sensor_index
        self.coef_ = None
        self.sensors_used = None
        self.indices_used = None