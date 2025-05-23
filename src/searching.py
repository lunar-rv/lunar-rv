import numpy as np
import random
import itertools
import json

# def evaluate_formula(traces, batch_size, operators, F_end=None, G_avg_end=None):
#     formula = FormulaFactory.build_tightest_formula(
#         operators=operators,
#         traces=traces,
#         F_end=F_end,
#         G_avg_end=G_avg_end,
#     )
#     rho = formula.evaluate(traces=traces, labels=False, return_arr=True).min(axis=1)
#     if "F" in operators:
#         r = -rho.ptp()
#         b = formula.f.end
#         score = contraction_fn(r, b, batch_size)
#     return score

with open('config.json', 'r') as file:
    config = json.load(file)

def grid_search(traces, batch_size, evaluation_fn, operators):
    best_params = None
    best_score = -np.inf
    bounded_operators = [op for op in operators if op != "G"]
    ranges = [range(1, batch_size) for _ in bounded_operators]
    for params in itertools.product(*ranges):
        args = (traces, batch_size, operators, *params)
        score = evaluation_fn(*args)
        if score > best_score:
            best_params = params
            best_score = score
    return best_params

def grid_search_1d(traces, batch_size, evaluation_fn, operators):
    best_score = -np.inf
    var_type = "F_end" if "F" in operators else "G_avg_end"
    for i in range(2, batch_size):
        if "F" in operators:
            score = evaluation_fn(traces, F_end=i, batch_size=batch_size, operators=operators)
        elif "G_avg" in operators:
            score = evaluation_fn(traces, G_avg_end=i, batch_size=batch_size, operators=operators)
        else:
            raise ValueError(f"Unrecognised 1D operators: {operators}")
        if score > best_score:
            best_end = i
            best_score = score
    return {var_type: best_end}


def hill_climbing_search(traces, batch_size, operators, evaluation_fn, max_iters=50):
    np.random.seed(42)
    bounded_operators = [op for op in operators if op != "G"]
    num_vars = len(bounded_operators)
    current_vars = np.random.randint(1, batch_size, size=num_vars)
    args = (traces, batch_size, operators, *current_vars)
    current_score = evaluation_fn(*args)
    
    for _ in range(max_iters):
        neighbours = []
        
        # Generate neighbours by incrementing/decrementing each variable
        for i in range(num_vars):
            if current_vars[i] > 1:
                neighbour = current_vars.copy()
                neighbour[i] -= 1
                neighbours.append(tuple(neighbour))
            if current_vars[i] < batch_size - 1:
                neighbour = current_vars.copy()
                neighbour[i] += 1
                neighbours.append(tuple(neighbour))
        next_vars = None
        next_score = current_score
        for vars_ in neighbours:
            args = (traces, batch_size, operators, *current_vars)
            score = evaluation_fn(*args)
            if score > next_score:
                next_vars = vars_
                next_score = score
        
        if next_vars is None:
            break
        current_vars = next_vars
        current_score = next_score
    return tuple(current_vars)

def simulated_annealing_search(traces, batch_size, operators, evaluation_fn, initial_temp=100, cooling_rate=0.95, max_iters=1000):
    np.random.seed(42)
    max_x = config["MAX_F_BOUND"]
    min_y = config["MIN_GBAR_BOUND"]
    current_x = np.random.randint(2, max_x)
    current_y = np.random.randint(min_y, batch_size)
    current_score = evaluation_fn(traces, batch_size, operators, current_x, current_y)
    
    best_x, best_y = current_x, current_y
    best_score = current_score
    
    temperature = initial_temp
    
    for _ in range(max_iters):
        next_x = current_x + random.choice([-1, 1])
        next_y = current_y + random.choice([-1, 1])
        next_x = np.clip(next_x, 2, max_x)
        next_y = np.clip(next_y, min_y, batch_size-1)
        
        next_score = evaluation_fn(traces, batch_size, operators, next_x, next_y)
        delta_score = next_score - current_score
        if delta_score > 0 or np.exp(delta_score / temperature) > np.random.rand():
            current_x, current_y = next_x, next_y
            current_score = next_score
            if current_score > best_score:
                best_x, best_y = current_x, current_y
                best_score = current_score
        temperature *= cooling_rate
        if temperature < 1e-3:
            break
    return best_x, best_y