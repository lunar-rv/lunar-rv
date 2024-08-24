import numpy as np
trace = np.genfromtxt("inputs/traces.csv", delimiter=",", dtype=str)
datetimes = trace[:50, -2:]
sensor_1 = np.arange(50)
sensor_2 = sensor_1 * 2
trace = np.column_stack((sensor_1, sensor_2, datetimes))
np.savetxt("inputs/test.csv", trace, delimiter=",", fmt="%s")