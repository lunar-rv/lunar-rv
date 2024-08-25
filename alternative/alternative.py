import numpy as np
# from sklearn.model_selection import train_test_split
from residuals_over_time import X_Y_split, cut
import matplotlib.pyplot as plt
from regressor import LargeWeightsRegressor

def get_residuals():
    pressure = np.genfromtxt("../inputs/pressures.csv", delimiter=",")
    sensors = cut(pressure, 96).reshape(-1, 96, 27)
    print(sensors.shape)
    train = sensors[:90]
    test = sensors[90:]
    all_residuals = []
    for i in range(27):
        print("SENSOR", i)
        X_train = np.delete(train, i, axis=2).reshape(-1, 26)
        y_train = train[:, :, i].reshape(-1)
        X_test = np.delete(test, i, axis=2).reshape(-1, 26)
        y_test = test[:, :, i].reshape(-1)
        model = LargeWeightsRegressor()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        residuals = predictions - y_test
        residuals = residuals.reshape(30, 96)
        all_residuals.append(residuals)
    np.save("residuals.npy", np.array(all_residuals))
    # plt.show()

def analyse():
    residuals = np.abs(np.load("residuals.npy"))
    mean_res = residuals.mean(axis=1)
    # plt.plot(mean_res.T)
    plt.plot(np.median(mean_res, axis=0))
    plt.show()


if __name__ == "__main__":
    analyse()