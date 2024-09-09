import numpy as np
def X_Y_split(data: np.ndarray, i: int, axis=1):
    index = [slice(None)] * data.ndim
    index[axis] = i
    Y = data[tuple(index)].astype(float)
    X = np.delete(data, i, axis=axis)
    return X, Y

def cut(data, dividend):
    return data[data.shape[0] % dividend:]