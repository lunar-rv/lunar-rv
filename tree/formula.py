print("Loading Formula class and subclasses...")

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
    def build_formula(boundary: float, operator="_", end: int = -1, sign="<=", last_residuals=None, last_raw_values=None):
        if operator == "_":
            return Basic(boundary, end=end, sign=sign)
        elif operator == "F":
            return Eventually(boundary, end=end, sign=sign, last_residuals=last_residuals, last_raw_values=last_raw_values)
        elif operator == "G":
            return Always(boundary, end=end, sign=sign)
        else:
            raise ValueError(f"Unknown operator: {operator}")

    @staticmethod
    def list_options(boundaries=[], binary=True):
        all_options = []
        def add_options(end, op, b=None):
            if binary:
                all_options.append((0, end, op))
            else:
                all_options.append((b, end, op))
        for end in range(1, config["BATCH_SIZE"]):
            if binary:
                add_options(end, "F")
            else:
                for b in boundaries:
                    add_options(end, "F", b)
        return all_options

    @abstractmethod
    def spec(self):
        pass


    def negate(self, verbose=True):
        if verbose:
            return "not " + self.spec
        else:
            return "¬" + self.stl_spec
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
    
    def evaluate2(self, traces, labels=True): # For testing purposes
        traces = traces[:, :-1].astype(float) if labels else traces
        end = self.end
        trace_end = traces.shape[1]
        for start in range(trace_end - end + 1):
            cut_traces = traces[:, start:start+end]
            values = self.boundary - np.min(cut_traces, axis=1).reshape(-1, 1)
            if start == 0:
                all_values = values
            else:
                all_values = np.hstack((all_values, values))
        return all_values.mean(axis=1)
    
class Eventually(BoundedFormula):
    def __init__(self, boundary, end: int, sign=">=", last_residuals=None, last_raw_values=None):
        super().__init__(boundary, sign, end)
        self.last_residuals = last_residuals
        self.last_raw_values = last_raw_values
    @property
    def spec(self):
        return f"eventually[0:{self.end})(error {self.sign} {self.boundary})"
    
    def evaluate_interval(self, traces):
        split_traces = traces.reshape(traces.shape[0], -1, self.end)
        values = self.boundary - np.min(split_traces, axis=2)
        return values

    
    def evaluate3(self, traces, labels=True): # For testing purposes
        traces = traces[:, :-1].astype(float) if labels else traces
        end = self.end
        trace_end = traces.shape[1]
        all_values = [None]
        for start in range(trace_end - end + 1):
            cut_traces = traces[:, start:start+end]
            values = self.boundary - np.min(cut_traces, axis=1).reshape(-1, 1)
            if start == 0:
                all_values = values
            else:
                all_values = np.hstack((all_values, values))
        return all_values
    
    def evaluate4(self, traces, labels=True, beta=10000): # Added beta as a parameter
        def log_min_approx(x, beta=10000):
            return - (1/beta) * np.log(np.sum(np.exp(-beta * x), axis=1))
        traces = traces[:, :-1].astype(float) if labels else traces
        end = self.end
        trace_end = traces.shape[1]
        all_values = [None]
        for start in range(trace_end - end + 1):
            cut_traces = traces[:, start:start+end]
            values = self.boundary - log_min_approx(cut_traces, beta).reshape(-1, 1)
            
            if start == 0:
                all_values = values
            else:
                all_values = np.hstack((all_values, values))
        
        return all_values

    def evaluate_single(self, trace, raw_values: np.ndarray, labels=True):
        raw_values = raw_values.reshape(1, -1)
        traces_arr = trace.reshape(1, -1)
        traces_arr = np.hstack((self.last_residuals, traces_arr)) if self.last_residuals is not None else traces_arr
        raw_values = np.hstack((self.last_raw_values, raw_values)) if self.last_raw_values is not None else raw_values
        evaluation = self.evaluate3(traces=traces_arr, labels=labels)[0]
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
    
    def human_readable(self):
        frequency = 60 // config["TIME_PERIOD"]
        hours = int(self.end // frequency)
        mins = config["TIME_PERIOD"] * (self.end % frequency)
        explanation = f"the value must be below {self.boundary} at some point in each period of {hours} hour(s) and {mins} minutes, or {self.end} consecutive readings"
        return explanation

class Always(Formula):
    def __init__(self, boundary, end: int = 1, sign=">="):
        super().__init__(boundary, sign=sign, end=end)

    @property
    def spec(self):
        return f"always (error {self.sign} {self.boundary})"
    
    def evaluate(self, traces):
        return self.boundary - traces.max()
    
    def human_readable(self):
        return f"The value must always be below {self.boundary}"
    
def main():
    # from sklearn.linear_model import LinearRegression
    # from sklearn.model_selection import train_test_split
    # traces = np.genfromtxt("inputs/preprocessed.csv", delimiter=",", dtype=float)
    # pressures = traces[:, :27]
    # def X_Y_split(data: np.ndarray, i: int):
    #     X = np.delete(data, i, axis=1)
    #     Y = data[:, i].astype(float)
    #     return X, Y
    # def cut(data, dividend):
    #     end = len(data) % dividend
    #     return data[:-end]
        
    # train, test = train_test_split(pressures, test_size=0.2)
    # X_train, y_train = X_Y_split(train, 0)
    # X_test, y_test = X_Y_split(test, 0)
    # model = LinearRegression()
    # model.fit(X_train, y_train)
    # predictions = model.predict(X_test)
    # residuals = np.abs(predictions - y_test) * 1000
    # residuals = cut(residuals, 96).reshape(-1, 96)
    long_formula = Formula.build_formula(0, "F", 4, "<=")
    import time
    residuals = np.random.rand(10000, 96)
    start = time.time()
    eval1 = long_formula.evaluate(residuals, labels=False)
    mid1 = time.time()
    eval2 = long_formula.evaluate2(residuals, labels=False)
    mid2 = time.time()
    eval3 = long_formula.evaluate3(residuals, labels=False)
    end = time.time()
    print("First:", mid1-start)
    print("Second:", mid2-mid1)
    print("Third:", end-mid2)
    # print(eval1)
    print("FIRST")
    print("Score:", eval1.ptp()) # LONG
    print("Max:", eval1.max())
    print("Min:", eval1.min())
    print("==========")
    # print(eval2)
    print("SECOND")
    print("Score:", eval2.ptp()) # SHORT
    print("Max:", eval2.max())
    print("Min:", eval2.min())
    print("==========")
    print("THIRD")
    print("Score:", eval3.ptp()) # DEQUE
    print("Max:", eval3.max())
    print("Min:", eval3.min())
    print("==========")

def X_Y_split(data: np.ndarray, i: int):
    X = np.delete(data, i, axis=1)
    Y = data[:, i].astype(float)
    return X, Y

def cut(data, dividend):
    end = len(data) % dividend
    return data[:-end]

contraction_fn = lambda r, size: r * (1 + np.exp(10 * (0.5 - size)))
def get_score(formula, residuals, i=-1):
    score = formula.evaluate3(residuals, labels=False)
    proportion = np.log(i) / np.log(96)
    score = -score.std()
    score = contraction_fn(score, proportion)
    return score

def plot_residuals():
    pressure_residuals = np.genfromtxt("inputs/pressure_residuals.csv", delimiter=",", dtype=float) * 1000
    anom_pressure_residuals = np.genfromtxt("inputs/anom_pressure_residuals.csv", delimiter=",", dtype=float) * 1000
    train = pressure_residuals[:47, :]
    test = pressure_residuals[47:, :]
    anom_test = anom_pressure_residuals[47:, :]#test + np.abs(np.random.normal(0, 0.02, test.shape))
    import matplotlib.pyplot as plt
    closeness_scores = []
    neg_scores = []
    pos_scores = []
    np.random.seed(0)
    for i in range(1, 96):
        formula = Formula.build_formula(0, "F", i, "<=")
        score = formula.evaluate3(train, labels=False)
        threshold = score.min()
        formula = Formula.build_formula(-threshold, "F", i, "<=")
        closeness_score = get_score(formula, train)
        neg_test_scores = formula.evaluate3(test, labels=False)
        pos_test_scores = formula.evaluate3(anom_test, labels=False)
        closeness_scores.append(closeness_score)
        # neg_scores.append(neg_test_scores.mean())
        # pos_scores.append(pos_test_scores.mean())
        neg_classifications = len(neg_test_scores[neg_test_scores < 0]) # classified as anomalous
        pos_classifications = len(pos_test_scores[pos_test_scores < 0]) # classified as anomalous
        neg_scores.append(neg_classifications)
        pos_scores.append(pos_classifications)
        print("Interval:", i, "Negative:", neg_classifications, "Positive:", pos_classifications)
    # plt.xlabel("Trace length")
    # plt.ylabel("Standard deviation of robustness values")
    # plt.plot(closeness_scores) 
    # plt.show()
    plt.title("Percentage of traces classified as anomalous")
    plt.xlabel("Interval length")
    plt.ylabel("Proportion of traces (%)")
    plt.plot(np.divide(neg_scores, 0.46), label="Safe")
    plt.plot(np.divide(pos_scores, 0.46), label="Anomalous")
    plt.legend(["Safe", "Anomalous"])
    plt.show()

def get_anomaly_bounds(indices) -> list:
    bounds = []
    N = len(indices)
    start_bound = None
    for i in range(N):
        this_value = indices[i]
        if i == 0 or indices[i-1] + 1 != this_value:
            start_bound = this_value - formula.end + 1
        if i+1 == N or indices[i+1] - 1 != this_value:
            bounds.append((start_bound, this_value))
    return bounds

if __name__ == "__main__":
    boundary = 0.025018738211056435
    formula = Formula.build_formula(boundary, "F", 24, "<=")
    trace = np.array([0.28547559247962817, 0.26462692391961834, 0.28547559247962817, 0.3063249211704179, 0.008126966558354848, 0.052574843004894734, 0.008126966558354848, 0.06882811599092514, 0.0031756988694703336, 0.08547719184467603, 0.0031756988694703336, 0.07912368403458978, 0.20283734279601118, 0.2059510238821248, 0.20283734279601118, 0.19972366170989755, 0.29949484326185705, 0.336089247907366, 0.29949484326185705, 0.2629003874166716, 0.4290392382291681, 0.5070291539974736, 0.4290392382291681, 0.3089858367679409, 0.14197785774081734, 0.09101568911105973, 0.14197785774081734, 0.192942402568147, 0.07308060791669929, 0.19059606368985668, 0.07308060791669929, 0.04443304068794257, 0.14604731202853827, 0.2007189769743338, 0.14604731202853827, 0.09137810532240664, 0.15927335689061572, 0.017395970448575576, 0.15927335689061572, 0.3011521549248844, 0.24963361718947918, 0.13444613551068588, 0.24963361718947918, 0.3648229060367117, 0.6427681625987315, 0.46247514145985813, 0.6427681625987315, 0.8230630910864046, 0.4092410020264184, 0.41902519699403906, 0.4092410020264184, 0.39945748503763046, 0.1133058573260376, 0.14649739708083934, 0.1133058573260376, 0.0801145267012339, 0.156290267218865, 0.013706081577545548, 0.156290267218865, 0.29887254551135695, 0.01189490346697672, 0.13108053474642986, 0.01189490346697672, 0.15486901213030216, 0.020105034374811454, 0.07203256631765562, 0.020105034374811454, 0.11224255302530461, 0.19832857972537643, 0.1232583367795359, 0.19832857972537643])
    evals = formula.evaluate3(trace.reshape(1, -1))
    anomaly_start_indices = np.where(evals[0] < 0)[0].tolist()
    print(anomaly_start_indices)
    print("Success points:", np.where(trace < boundary)[0].tolist())

    exit()
    bounds = get_anomaly_bounds(anomaly_start_indices)
    print(bounds)
    # for trace in traces:
    #     score = formula.evaluate_single(np.array(trace), labels=False)
    #     print(score)