print("Loading formula synthesis functions...")
from searching import hill_climbing_search, grid_search_1d
from tree.new_formula import FormulaFactory
import numpy as np
import json
with open("config.json") as config_file:
    config = json.load(config_file)

# FIX THE CLOSENESS FUNCTION.
contraction_fn = lambda r, b, max_size: r * np.exp((0.5 * b / max_size)-0.5)

def evaluate_formula(traces, batch_size, operators, F_end=None, G_avg_end=None):
    formula = FormulaFactory.build_tightest_formula(
        operators=operators,
        traces=traces,
        F_end=F_end,
        G_avg_end=G_avg_end,
    )
    rho = formula.evaluate(traces=traces, labels=False, return_arr=True).min(axis=1)
    if "F" in operators:
        r = -rho.ptp()
        b = formula.f.end
        score = contraction_fn(r, b, batch_size)
    return score

def positive_synth(traces, operators, prev_formula=None):
    best_formula = None
    batch_size = traces.size // traces.shape[0]
    bounded_operators = [op for op in operators if op != "G"]
    best_formula_kwargs = {
        "operators": operators,
        "traces": traces,
        "F_end": -1,
        "G_avg_end": -1
    }
    if len(bounded_operators) == 1:
        best_end = grid_search_1d(traces, batch_size, evaluation_fn=evaluate_formula, operators=operators)
        best_formula_kwargs.update(best_end)
    else:  
        best_x, best_y = hill_climbing_search(traces=traces, batch_size=batch_size, evaluation_fn=evaluate_formula, operators=operators)
        best_formula_kwargs.update({
            "F_end": best_x,
            "G_avg_end": best_y
        })
    best_formula = FormulaFactory.build_tightest_formula(**best_formula_kwargs)
    if prev_formula:
        lrv = prev_formula.last_raw_values
        lr = prev_formula.last_residuals
        if prev_formula.max_length > best_formula.max_length:
            lrv = lrv[1-best_formula.max_length:]
            lr = lr[1-best_formula.max_length:]
        elif prev_formula.max_length < best_formula.max_length:
            padding_length = best_formula.max_length - prev_formula.max_length
            lrv = np.pad(lrv, ((0, 0), (padding_length, 0)), mode='constant', constant_values=np.nan)
            lr = np.pad(lr, ((0, 0), (padding_length, 0)), mode='constant', constant_values=np.nan)
        best_formula.last_raw_values = lrv
        best_formula.last_residuals = lr
    return best_formula

def main():
    import telex.synth as tx
    import time
    from formula import Formula
    import matplotlib.pyplot as plt
    traces = np.genfromtxt("inputs/pressure_residuals.csv", delimiter=",", dtype=float)
    traces = traces[:2, :]
    traces *= 1000
    # print(traces.shape)
    formula1 = positive_synth(traces)
    formula2 = Formula.build_formula(0.01247759930412972, "F", 3, "<=")
    eval1 = formula1.evaluate3(traces, labels=False).flatten()
    eval2 = formula2.evaluate3(traces, labels=False).flatten()
    plt.plot(eval1, label="New positive synthesis")
    plt.plot(eval2, label="TeLEx")
    plt.legend()
    plt.title("Robustness of synthesised formulae on pressure residual traces over 2 days")
    plt.xlabel("Time")
    plt.ylabel("Robustness")
    plt.show()
    exit()

if __name__ == "__main__":
    main()