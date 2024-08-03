from abc import ABC, abstractmethod
import numpy as np
import json
with open("config.json") as config_file:
    config = json.load(config_file)

class Formula:
    def __init__(self, boundary: float, sign=">=", end=None):
        self.boundary = boundary
        self.sign = sign
        self.end = end

    @property
    def stl_spec(self):
        spec = self.spec
        spec = spec.replace("eventually", "F")
        spec = spec.replace("always", "G")
        spec = spec.replace(" s", "")
        return spec
    
    def __repr__(self):
        return self.spec

    @staticmethod
    def build_formula(boundary: float, operator="_", end: int = -1, sign="<="):
        if operator == "_":
            return Basic(boundary, end=end, sign=sign)
        elif operator == "F":
            return Eventually(boundary, end=end, sign=sign)
        elif operator == "G":
            return Always(boundary, end=end, sign=sign)
        else:
            raise ValueError(f"Unknown operator: {operator}")

    @staticmethod
    def list_options(operators="FG_", boundaries=None, binary=True, invariance=False, use_mean=True):
        if invariance:
            if use_mean:
                return [(b, 1, "G") for b in boundaries]
            else:
                return [(b, 96, "G") for b in boundaries]
        all_options = []
        time_period = config["BATCH_SIZE"]
        def add_options(end, op, b=None):
            if binary:
                all_options.append((0, end, op))
            else:
                all_options.append((b, end, op))
        for op in operators:
            if op in "FG":
                end = 1
                while time_period % end == 0 and time_period > end:
                    if binary:
                        add_options(end, op)
                    else:
                        for b in boundaries:
                            add_options(end, op, b)
                    end *= 2
            else:
                if binary:
                    add_options(1, op)
                else:
                    for b in boundaries:
                        add_options(1, op, b)
        return all_options

    @abstractmethod
    def spec(self):
        pass


    def negate(self, verbose=True):
        if verbose:
            return "not " + self.spec
        else:
            return "¬" + self.stl_spec
    # def evaluate(self, trace, labels=True):
    #     values = trace[:-1].astype(float) if labels else trace
    #     rtamt_trace = {"time": range(len(values)), "P": values}
    #     rtamt_spec.spec = self.spec
    #     try:
    #         rtamt_spec.parse()
    #     except rtamt.exception.exception.RTAMTException as e:
    #         print(e)
    #         print("Spec was:", self.spec)
    #     output = rtamt_spec.evaluate(rtamt_trace)
    #     robustness = np.array(output)[:, -1]
    #     return np.mean(robustness)
    @abstractmethod
    def evaluate(self, trace, labels=True):
        pass


class Basic(Formula):
    def __init__(self, boundary, sign=">=", end=None):
        super().__init__(boundary, sign, end)

    @property
    def spec(self):
        return f"error {self.sign} {self.boundary}"

    def negate(self, verbose=True):
        negated_sign = ""
        if self.sign == ">=":
            negated_sign = "<"
        elif self.sign == "<=":
            negated_sign = ">"
        elif self.sign == ">":
            negated_sign = "<="
        elif self.sign == "<":
            negated_sign = ">="
        return f"error {negated_sign} {self.boundary}"
    
    def evaluate(self, traces, labels=True):
        traces = traces[:, :-1].astype(float) if labels else traces
        return np.mean(self.boundary - traces, axis=1)


class BoundedFormula(Formula, ABC):
    def __init__(self, boundary, sign=">=", end=None):
        super().__init__(boundary, sign, end)
    @abstractmethod
    def evaluate_interval(self, traces):
        pass
    def evaluate(self, traces, labels=True):
        traces = traces[:, :-1].astype(float) if labels else traces
        values = self.evaluate_interval(traces)
        for i in range(1, self.end):
            j = self.end - i
            cut_traces = traces[:, i:-j]
            values = np.hstack((values, self.evaluate_interval(cut_traces)))
        return values.mean(axis=1)

class Eventually(BoundedFormula):
    def __init__(self, boundary, end: int, sign=">="):
        super().__init__(boundary, sign, end)

    @property
    def spec(self):
        return f"eventually[0:{self.end}](error {self.sign} {self.boundary})"
    
    def evaluate_interval(self, traces):
        split_traces = traces.reshape(traces.shape[0], -1, self.end)
        values = self.boundary - np.min(split_traces, axis=2)
        return values

class Always(BoundedFormula):
    def __init__(self, boundary, end: int, sign=">="):
        super().__init__(boundary, sign=sign, end=end)

    @property
    def spec(self):
        if self.end == 1 and config["USE_MEAN"]:
            return f"mean error {self.sign} {self.boundary}"
        if self.end == 96 and not config["USE_MEAN"]:
            return f"always (error {self.sign} {self.boundary})"
        return f"always[0:{self.end} s](error {self.sign} {self.boundary})"
    
    def evaluate_interval(self, traces):
        split_traces = traces.reshape(traces.shape[0], -1, self.end)
        values = self.boundary - np.max(split_traces, axis=2)
        return values
    
def main():
    import random
    safe_traces = np.array([[random.gauss(2, 1) for _ in range(100)] for _ in range(5)])
    formula = Formula.build_formula(1.5, "F", 3)
    print(formula.evaluate(safe_traces))
if __name__ == "__main__":

    # main()
    class A:
        def __init__(self):
            self.a = 1
    class B(A):
        def __init__(self):
            self.b = 2
    class C(B):
        def __init__(self):
            super.super().__init__()
            self.c = 3
    c = C()
    print(c.a)