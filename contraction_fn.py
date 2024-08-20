import numpy as np
import matplotlib.pyplot as plt
from formula import Formula
def contraction_fn(r, size):
    exp = np.exp(5 * (size-0.5))
    print(r, exp,  r * exp)
    return r * exp   

residuals = np.genfromtxt("inputs/pressure_residuals.csv", delimiter=",", dtype=float)
first = residuals[0]
evaluations = []
tightnesses = []
# log_evaluations = []
for i in range(1, 96):
    formula = Formula.build_formula(0.001, "F", i, "<=")
    r = formula.evaluate4(residuals, labels=False).min(axis=1)
    ptp = -r.ptp()
    tightnesses.append(ptp)
    evaluations.append(contraction_fn(ptp, i / 96))
# plt.plot(first, label="Original")
# plt.plot(evaluations, label="Tightness score before contraction function")
plt.plot(evaluations, label="Overall score")
plt.legend()
plt.show()