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
                return [(b, config["BATCH_SIZE"], "G") for b in boundaries]
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
    def __init__(self, boundary, end: int, sign=">="):
        super().__init__(boundary, sign, end)
        self.last = None
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
    
    def evaluate_single(self, trace, labels=True):
        traces_arr = trace.reshape(1, -1)
        traces_arr = np.hstack((self.last, traces_arr)) if self.last is not None else traces_arr
        evaluation = self.evaluate3(traces=traces_arr, labels=labels)
        if self.last is None or self.end <= traces_arr.shape[1]:
            self.last = traces_arr[:, -self.end + 1:]
        else:
            self.last = traces_arr
        return evaluation[0]
    
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
    # score = contraction_fn(score, proportion)
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
    formula = Formula.build_formula(0.03389871128318955, "F", 22, "<=")
    trace = np.array([0.0425063702280612, 0.18826705324576434, 0.4012793802426924, 0.524821307393307, 0.4012793802426924, 0.2777362156046216, 0.08972943046809193, 0.2187346138168532, 0.08972943046809193, 0.03927549130232355, 0.07830282062214547, 0.20914935263380632, 0.07830282062214547, 0.05254557409208396, 0.2529852132780244, 0.2985415138590797, 0.2529852132780244, 0.20742634423103617, 0.04664647645037656, 0.18886567125377465, 0.04664647645037656, 0.08322846402686324, 0.026415579590776533, 0.04456207795779099, 0.026415579590776533, 0.097391329790468, 0.14059696562890647, 0.056135059450304614, 0.14059696562890647, 0.22505910966661574, 0.034562718659123665, 0.07952086098149402, 0.034562718659123665, 0.010397645803916084, 0.08506007625892764, 0.19154417566016185, 0.08506007625892764, 0.021425930491116713, 0.05393736400750859, 0.2951377141758929, 0.05393736400750859, 0.1872615554605328, 0.1230854419233332, 0.132138942095468, 0.1230854419233332, 0.11403031783788556, 0.05873009340146701, 0.025932757779956184, 0.05873009340146701, 0.14339103723418067, 0.08308112943081203, 0.07810169207243464, 0.08308112943081203, 0.08806014353105254, 0.07118642531960331, 0.10066471416741996, 0.07118642531960331, 0.04170941405984632, 0.17670088049139693, 0.21716866414733804, 0.17670088049139693, 0.13623405963153132, 0.010912479897609573, 0.12116227758785653, 0.010912479897609573, 0.09933414142337949, 0.07886698900990904, 0.1203811029106272, 0.07886698900990904, 0.0373523756720083, 0.19994758761238476, 0.18341525718406845, 0.19994758761238476, 0.21648105580612462, 0.00836433863940153, 0.07948559696224419, 0.00836433863940153, 0.06275466600965535, 0.047952120222787, 0.03685606157304358, 0.047952120222787, 0.05904706296424797, 0.08613621388102455, 0.13012446222874022, 0.08613621388102455, 0.042148118645540106, 0.2290945076316095, 0.23088366088390086, 0.2290945076316095, 0.2273081929909833, 0.22096992196038445, 0.1937335330804632, 0.22096992196038445, 0.2482059645152017, 0.0425063702280612, 0.1032554733440133, 0.0425063702280612, 0.18826705324576434, 0.4012793802426924, 0.524821307393307, 0.4012793802426924, 0.2777362156046216, 0.08972943046809193, 0.2187346138168532, 0.08972943046809193, 0.03927549130232355, 0.07830282062214547, 0.20914935263380632, 0.07830282062214547, 0.05254557409208396, 0.2529852132780244, 0.2985415138590797, 0.2529852132780244, 0.20742634423103617, 0.04664647645037656, 0.18886567125377465, 0.04664647645037656])
    evals = formula.evaluate_single(trace)
    anomaly_start_indices = np.where(evals < 0)[0].tolist()
    bounds = get_anomaly_bounds(anomaly_start_indices)
    print(bounds)
    # for trace in traces:
    #     score = formula.evaluate_single(np.array(trace), labels=False)
    #     print(score)