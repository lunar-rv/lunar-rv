import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def X_Y_split(data: np.ndarray, i: int):
    X = np.delete(data, i, axis=1)
    Y = data[:, i].astype(float)
    return X, Y
def cut(data, dividend):
    end = len(data) % dividend
    return data[:-end]
pressures = np.genfromtxt("inputs/pressures.csv", delimiter=",", dtype=float)
train, test = train_test_split(pressures, test_size=0.5, random_state=0)
all_residuals = []
model = LinearRegression()
for i in range(1):
    print("SENSOR", i)
    X_train, y_train = X_Y_split(train, i)
    X_test, y_test = X_Y_split(test, i)
    probabilities = np.random.choice([0, 1], size=y_test.shape, p=[0, 1])
    anomalies = probabilities * 1 * 2e-4 # * np.abs(np.random.normal(0, 1, y_test.shape))
    y_test += anomalies
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    residuals = np.abs(predictions - y_test)
    residuals = cut(residuals, 96).reshape(-1, 96)
    mean_res = residuals.mean(axis=0)
    all_residuals.append(residuals) # mean_res)
residuals = np.array(all_residuals[0])
np.savetxt("inputs/anom_pressure_residuals.csv", residuals, delimiter=",")
exit()
print(residuals.shape)
mean_res = residuals.mean(axis=0)
plt.plot(mean_res)
plt.title("Mean Absolute Residuals for Temperature Sensors over 1 day")
plt.xlabel("Time of day (hours)")
plt.ylabel("Mean Absolute Residuals (K)")
ticks = np.linspace(0, 96, 9)
labels = [f"{int(tick) // 4}:00" for tick in ticks]
labels[-1] = "0:00"
plt.xticks(ticks, labels=labels)
plt.show()
day = np.arange(24, 72)
night = np.delete(np.arange(96), day)
day_values = residuals[:, day]
night_values = residuals[:, night]
day_min = np.min(day_values, axis=1)
night_min = np.min(night_values, axis=1)
print("DAY")
print(day_min.ptp())
print(day_min.std())
print("NIGHT")
print(night_min.ptp())
print(night_min.std())
