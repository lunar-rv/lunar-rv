import numpy as np
trace = np.genfromtxt("inputs/traces.csv", delimiter=",", dtype=str)
datetimes = trace[:50, -2:]
sensor_1 = np.arange(50)
sensor_2 = sensor_1 * 2
sensor_3 = np.arange(50)
sensor_4 = sensor_3 * 2 + np.random.normal(0, 0.01, 50)
sensor_5 = sensor_3 * 0.5 + sensor_4 * 0.6 + np.random.normal(0, 0.01, 50)
trace = np.column_stack((sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, datetimes))
np.savetxt("inputs/test.csv", trace, delimiter=",", fmt="%s")