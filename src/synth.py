print("Loading formula synthesis functions...")
import searching
from tree.formula import FormulaFactory
import numpy as np
import json
with open("config.json") as config_file:
    config = json.load(config_file)

contraction_fn = lambda r, b, max_size: r * np.exp((0.5 * b / max_size)-0.5)

def evaluate_formula(traces, batch_size, operators, F_end=None, G_avg_end=None):
    formula = FormulaFactory.build_tightest_formula(
        operators=operators,
        traces=traces,
        F_end=F_end,
        G_avg_end=G_avg_end,
    )
    rho = formula.evaluate(traces=traces, labels=False, return_arr=True).min(axis=1)
    score = -rho.ptp()
    if "F" in operators:
        b = formula.f.end
        score = contraction_fn(score, b, batch_size)
    return score

def positive_synth(traces, operators, prev_formula=None):
    best_formula = None
    batch_size = traces.shape[1]
    bounded_operators = [op for op in operators if op != "G"]
    best_formula_kwargs = {
        "operators": operators,
        "traces": traces,
        "F_end": -1,
        "G_avg_end": -1
    }
    if len(bounded_operators) == 1:
        best_end = searching.grid_search_1d(traces, batch_size, evaluation_fn=evaluate_formula, operators=operators)
        best_formula_kwargs.update(best_end)
    elif len(bounded_operators) > 1:
        best_x, best_y = searching.simulated_annealing_search(traces=traces, batch_size=batch_size, evaluation_fn=evaluate_formula, operators=operators)
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
            if lr is not None and lrv is not None:
                lrv = np.pad(lrv, ((0, 0), (padding_length, 0)), mode='constant', constant_values=np.nan)
                lr = np.pad(lr, ((0, 0), (padding_length, 0)), mode='constant', constant_values=0)
        best_formula.last_raw_values = lrv
        best_formula.last_residuals = lr
    return best_formula