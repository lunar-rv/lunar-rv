import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import json
with open("config.json") as f:
    config = json.load(f)

contraction_fn = lambda r, b, max_size: r * np.exp(5 * (b / max_size)-0.5)

class Predicate(ABC):
    def __init__(self, boundary, end):
        self.boundary = boundary
        self.end = end
    @abstractmethod
    def evaluate(self, values):
        pass
    @abstractmethod
    def __repr__(self):
        pass
        
class F(Predicate):
    def __init__(self, boundary, end):
        super().__init__(boundary, end)
    def evaluate(self, traces, labels=False):
        traces = traces[:, :-1].astype(float) if labels else traces
        trace_end = traces.shape[1]
        all_values = [None]
        for start in range(trace_end - self.end + 1):
            cut_traces = traces[:, start:start+self.end]
            values = self.boundary - np.min(cut_traces, axis=1).reshape(-1, 1)
            if start == 0:
                all_values = values
            else:
                all_values = np.hstack((all_values, values))
        return all_values
    def __repr__(self):
        return f"eventually[0, {self.end}) error <= {self.boundary})"
class G_avg(Predicate):
    def __init__(self, boundary, end):
        super().__init__(boundary, end)
    def __repr__(self):
        return f"mean[0, {self.end}) error <= {self.boundary})"
    def evaluate(self, traces, labels=False):
        traces = traces[:, :-1].astype(float) if labels else traces
        trace_end = traces.shape[1]
        all_values = [None]
        for start in range(trace_end - self.end + 1):
            cut_traces = traces[:, start:start+self.end]
            values = self.boundary - np.mean(cut_traces, axis=1).reshape(-1, 1)
            if start == 0:
                all_values = values
            else:
                all_values = np.hstack((all_values, values))
        return all_values
class G(Predicate):
    def __init__(self, boundary, end=None):
        super().__init__(boundary, None)
    def __repr__(self):
        return f"always error <= {self.boundary})"
    def evaluate(self, traces, labels=False):
        traces = traces[:, :-1].astype(float) if labels else traces
        values = self.boundary - traces.max(axis=1).reshape(-1, 1)
        return values
    
class Formula:
    def __init__(self, g: G, f: F, g_avg: G_avg):
        self.g = g
        self.f = f
        self.g_avg = g_avg
        self.epsilon = config["EPSILON"]
        self.last_residuals = None
        self.last_raw_values = None
    def __repr__(self):
        return f"({self.g}) and ({self.f}) and ({self.g_avg})"
    @staticmethod
    def build_formula(traces: np.ndarray, F_end: int, G_avg_end: int):
        def get_mu(phi_0, traces):            
            rho_0 = phi_0.evaluate(traces)
            rho_crit = rho_0.min()
            mu = config["EPSILON"] - rho_crit
            return mu
        f_0 = F(boundary=0, end=F_end)
        g_avg_0 = G_avg(boundary=0, end=G_avg_end)
        g_0 = G(boundary=0)
        mu_f = get_mu(f_0, traces)
        mu_g_avg = get_mu(g_avg_0, traces)
        mu_g = get_mu(g_0, traces)
        f = F(boundary=mu_f, end=F_end)
        g_avg = G_avg(boundary=mu_g_avg, end=G_avg_end)
        g = G(boundary=mu_g)
        return Formula(g, f, g_avg)

    def evaluate(self, traces, labels=False):
        g_eval = self.g.evaluate(traces, labels)
        f_eval = self.f.evaluate(traces, labels)
        g_avg_eval = self.g_avg.evaluate(traces, labels)
        all_evals = np.hstack((g_eval, f_eval, g_avg_eval))
        return all_evals.min(axis=1)
    
    def evaluate_single(self, trace, raw_values: np.ndarray, labels=True):
        raw_values = raw_values.reshape(1, -1)
        traces_arr = trace.reshape(1, -1)
        traces_arr = np.hstack((self.last_residuals, traces_arr)) if self.last_residuals is not None else traces_arr
        raw_values = np.hstack((self.last_raw_values, raw_values)) if self.last_raw_values is not None else raw_values
        evaluation = self.evaluate(traces_arr, labels)[0]
        if self.last_residuals is None or self.end <= traces_arr.shape[1]:
            if self.end != 1:
                self.last_residuals = traces_arr[:, -self.end + 1:]
                self.last_raw_values = raw_values[:, -self.end + 1:]
            else:
                self.last_residuals = None
                self.last_raw_values = None
        else:
            self.last_residuals = traces_arr
            self.last_raw_values = raw_values
        return evaluation

def main():
    traces = np.genfromtxt('inputs/pressure_residuals.csv', delimiter=',', dtype=float)
    F_eval = F(end=12, boundary=9.000768419690358e-05).evaluate(traces)
    G_avg_eval = G_avg(end=12, boundary=0.00027961570794639116).evaluate(traces)
    G_eval = G(boundary=0.0009378794303354816).evaluate(traces)
    G_avg_rho_crit = G_avg_eval.min(axis=1)
    F_rho_crit = F_eval.min(axis=1)
    formula = Formula.build_formula(
        traces=traces,
        F_end=12,
        G_avg_end=12,
    )
    formula_eval = formula.evaluate(traces, labels=False)
    print(formula)
    # score = formula_eval.ptp()
    # plt.plot(G_avg_rho_crit, label="G_avg")
    # plt.plot(F_rho_crit, label="F")
    # plt.plot(G_eval, label="G")
    # # plt.plot(formula_eval, label="Formula")
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()