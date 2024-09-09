import numpy as np
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
    @abstractmethod
    def human_readable(self, time_period):
        pass
    @property
    def spec(self):
        return repr(self)
    def negate(self):
        return f"¬({self.spec})"
class F(Predicate):
    def __init__(self, boundary, end):
        super().__init__(boundary, end)
    def evaluate(self, traces, labels=False):
        traces = traces[:, :-1].astype(float) if labels else traces
        traces = np.abs(traces)
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
        return f"eventually[0, {self.end}) error <= {self.boundary}"
    def human_readable(self, time_period):
        frequency = 60 // time_period
        hours = int(self.end // frequency)
        mins = time_period * (self.end % frequency)
        explanation = f"the value must be below {self.boundary} at some point in each period of {hours} hour(s) and {mins} minutes, or {self.end} consecutive readings"
        return explanation
class G_avg(Predicate):
    def __init__(self, boundary, end):
        super().__init__(boundary, end)
    def __repr__(self):
        return f"mean[0, {self.end}) error <= {self.boundary}"
    def evaluate(self, traces, labels=False):
        traces = traces[:, :-1].astype(float) if labels else traces
        traces = np.abs(traces)
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
    def human_readable(self, time_period):
        frequency = 60 // time_period
        hours = int(self.end // frequency)
        mins = time_period * (self.end % frequency)
        return f"The mean error must be below {self.boundary} in each period of {hours} hour(s) and {mins} minutes, or {self.end} consecutive readings"
class G(Predicate):
    def __init__(self, boundary, end=None):
        super().__init__(boundary, None)
    def __repr__(self):
        return f"error <= {self.boundary}"
    def human_readable(self, time_period):
        return f"The error must always be below {self.boundary}"
    def evaluate(self, traces, labels=False):
        traces = traces[:, :-1].astype(float) if labels else traces
        traces = np.abs(traces)
        values = self.boundary - traces.max(axis=1).reshape(-1, 1)
        return values
    
class Formula:
    def __init__(self, g: G = None, f: F = None, g_avg: G_avg = None):
        self.g = g
        self.f = f
        self.g_avg = g_avg
        self.last_residuals = None
        self.last_raw_values = None
    def __repr__(self):
        formulae = [f"{repr(phi)}" for phi in self.__list__() if phi]
        return "always[0, batch_size) ((" + ') and ('.join(formulae) + ")"
    def __getitem__(self, index):
        return self.__list__()[index]
    def __list__(self):
        return [phi for phi in (self.g, self.f, self.g_avg) if phi is not None]
    @property
    def max_length(self):
        if self.f is not None and self.g_avg is not None:
            return max(self.g_avg.end, self.f.end)
        if self.f is not None:
            return self.f.end
        if self.g_avg is not None:
            return self.g_avg.end
        else:
            return 1000
    def human_readable(self, time_period):
        return "".join([f"\n\t- {phi.human_readable(time_period)}" for phi in self.__list__()])
    def evaluate(self, traces, labels=False, return_arr=False) -> np.ndarray:
        if not return_arr:
            return {phi: phi.evaluate(traces, labels) for phi in self.__list__()}
        else:
            return np.hstack([phi.evaluate(traces, labels) for phi in self.__list__()])
    def only_global(self):
        return len(list(self)) == 1 and list(self)[0] == self.g
    def evaluate_single(self, trace, raw_values: np.ndarray, labels=True, return_arr=False):
        raw_values = raw_values.reshape(1, -1)
        traces_arr = trace.reshape(1, -1)
        traces_arr = np.hstack((self.last_residuals, traces_arr)) if self.last_residuals is not None else traces_arr
        raw_values = np.hstack((self.last_raw_values, raw_values)) if self.last_raw_values is not None else raw_values
        if self.last_residuals is None or self.max_length <= traces_arr.shape[1]:
            if self.max_length != 1 and not self.only_global():
                self.last_residuals = traces_arr[:, -self.max_length + 1:]
                self.last_raw_values = raw_values[:, -self.max_length + 1:]
            else:
                self.last_residuals = None
                self.last_raw_values = None
        else:
            self.last_residuals = traces_arr
            self.last_raw_values = raw_values
        if return_arr:
            evaluation = self.evaluate(np.abs(traces_arr), labels, return_arr=True)[0]
            return evaluation
        else:
            return self.evaluate(np.abs(traces_arr), labels, return_arr=False)
        
class FormulaFactory:
    @staticmethod
    def build_tightest_formula(traces: np.ndarray, operators: list, F_end: int = -1, G_avg_end: int = -1):
        G_end = None
        epsilon = config["EPSILON_COEF"] * np.std(traces) + 1e-8
        def get_mu(phi_0, traces):
            rho_0 = phi_0.evaluate(traces)
            rho_crit = rho_0.min()
            mu = epsilon - rho_crit
            return mu
        kwargs = {}
        for op in operators:
            end = locals()[f"{op}_end"]
            Class = globals()[op]
            phi_0 = Class(boundary=0, end=end)
            mu = get_mu(phi_0, traces)
            kwargs[op.lower()] = Class(boundary=mu, end=end)
        return Formula(**kwargs)
    
    @staticmethod
    def build_formula(operator, end, boundary) -> Predicate:
        if operator == "F":
            return F(boundary, end)
        elif operator == "G_avg":
            return G_avg(boundary, end)
        elif operator == "G":
            return G(boundary)
        else:
            raise ValueError(f"Invalid operator {operator}")
        
    @staticmethod
    def list_options(boundaries, operators, binary=True, batch_size=96):
        all_options = []
        def add_options(end, op, b=None):
            if binary:
                all_options.append((0, end, op))
            else:
                all_options.append((b, end, op))
        if "G" in operators:
            if binary:
                add_options(None, "G")
            else:
                for b in boundaries:
                    add_options(None, "G", b)
        bounded_operators = [phi for phi in operators if phi != "G"]
        for operator in bounded_operators:
            for end in range(1, batch_size):
                if binary:
                    add_options(end, operator)
                else:
                    for b in boundaries:
                        add_options(end, operator, b)
        return all_options