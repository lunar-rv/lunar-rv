import numpy as np
indices = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 44, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]
prev_backlog_size = 93
end = 92
def get_anomaly_bounds(indices) -> list:
    bounds = []
    N = len(indices)
    start_bound = None
    for i in range(N):
        this_value = indices[i]
        if i == 0 or indices[i-1] + 1 != this_value:
            start_bound = this_value - prev_backlog_size
        if i+1 == N or indices[i+1] - 1 != this_value:
            bounds.append((start_bound, this_value + end - prev_backlog_size - 1))
    return bounds
bounds = get_anomaly_bounds(indices)
bounds = np.array(bounds) + prev_backlog_size
print(bounds)