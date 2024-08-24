lists = {
    "a": [1,2,3,4,5],
    "b": [2,4,45,6,-2],
    "c": [3,4,5,6,7],
}
import numpy as np

a = np.array(list((lists.values())))
b = a.std(axis=0) * 0.1

