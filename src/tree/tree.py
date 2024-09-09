print("Loading code for decision tree construction...")
import numpy as np
from tree.new_formula import FormulaFactory
import random
import json
with open("config.json") as file:
    config = json.load(file)
tree_config = config["TREE_CONFIG"]

def entropy(left_lab, right_lab) -> float:
    def calculate_entropy(labels):
        if len(labels) == 0:
            return 0
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    total_count = len(left_lab) + len(right_lab)
    if total_count == 0:
        return 0
    left_weight = len(left_lab) / total_count
    right_weight = len(right_lab) / total_count
    left_entropy = calculate_entropy(left_lab)
    right_entropy = calculate_entropy(right_lab)
    total_entropy = left_weight * left_entropy + right_weight * right_entropy
    return total_entropy

def stl_entropy(left_lab, left_rob, right_lab, right_rob) -> float:
    total_size = len(left_rob) + len(right_rob)
    l = len(left_rob) / total_size
    r = len(right_rob) / total_size

    def calculate_entropy(lab, rob):
        if lab.size == 0:
            return 0
        abs_rob = np.abs(rob)
        classes = np.unique(lab)
        classified_rob_sums = dict.fromkeys(classes, 0)
        for i in range(lab.size):
            l = lab[i]
            if abs_rob[i] == 0:
                abs_rob[i] = 1e-6 # To avoid division by zero
            classified_rob_sums[l] += abs_rob[i]
        rob_sums = list(classified_rob_sums.values())
        probabilities = rob_sums / np.sum(rob_sums)
        entropy_value = -np.sum(probabilities * np.log(probabilities))
        return entropy_value

    H_left = calculate_entropy(left_lab, left_rob)
    H_right = calculate_entropy(right_lab, right_rob)
    H = l * H_left * r * H_right
    return H

def choose_majority(labels) -> str:
    values, counts = np.unique(labels, return_counts=True)
    choice = values[np.argmax(counts)]
    return choice

def find_best_binary_threshold(values, labels, n=5) -> float:
    safe_values = values[labels == "Safe"]
    anomaly_values = values[labels == "Anomaly"]

    best_threshold = (safe_values.min() + anomaly_values.max()) / 2
    return best_threshold


def split_with_formula(traces: np.ndarray, formula, return_traces=False, binary=True) -> tuple:
    try:
        evaluations = formula.evaluate(traces, labels=True).min(axis=1)
    except:
        print("Error evaluating formula", formula)
        print("Traces:", traces)
        exit()
    if return_traces:
        left_trace = traces[evaluations > 0]
        right_trace = traces[evaluations <= 0]
        return left_trace, right_trace
    labels = traces[:, -1]
    if binary:
        threshold = find_best_binary_threshold(evaluations, labels, n=10)
        epsilon = 1e-8
        formula.boundary = epsilon - threshold # For floating point arithmetic errors where value is close to 0
        evaluations = formula.evaluate(traces, labels=True).min(axis=1)
    left_lab = labels[evaluations > 0]
    left_rob = evaluations[evaluations > 0]
    right_lab = labels[evaluations <= 0]
    right_rob = evaluations[evaluations <= 0]
    return left_lab, left_rob, right_lab, right_rob


def choose_formula(traces: np.ndarray, batch_size, operators: list, binary=False):
    best_entropy = np.inf
    best_formula = None
    values = traces[:, :-1].astype(float)
    boundaries = np.linspace(values.min(), values.max(), num=6)
    np.random.shuffle(boundaries)
    for boundary, end, operator in FormulaFactory.list_options(binary=binary, boundaries=boundaries, batch_size=batch_size, operators=operators):
        formula = FormulaFactory.build_formula(boundary=boundary, end=end, operator=operator)
        left_lab, left_rob, right_lab, right_rob = split_with_formula(traces, formula, binary=binary)
        H_1 = stl_entropy(left_lab, left_rob, right_lab, right_rob)
        H_2 = entropy(left_lab, right_lab)
        beta = tree_config["BETA"]
        # beta = 5
        epsilon = 1e-7
        # F_beta = (1 + beta ** 2) * H_1 * H_2 / ((beta ** 2 * H_1 + H_2) + epsilon)
        F_beta = H_1 + H_2 / beta
        if F_beta < best_entropy:
            best_formula = formula
            best_entropy = F_beta
    assert best_formula is not None
    return best_formula


class TreeNode:
    traces: np.ndarray
    formula: str
    max_depth: int
    def __init__(self, left, right, traces, formula: str, value=None, max_depth=tree_config["MAX_DEPTH"]):
        self.left = left
        self.right = right
        self.traces = traces
        self.formula = formula
        self.value = value
        self.max_depth = max_depth

    @property
    def labels(self) -> np.ndarray:
        return self.traces[:, -1]

    def count_labels(self) -> dict:
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        label_counts = {v: c for v, c in zip(unique_labels, counts)}
        return label_counts

    def __repr__(self) -> str:
        label_counts = self.count_labels()
        if self.left or self.right:
            return f"Node(formula='{self.formula.spec}', labels={label_counts})"
        else:
            return f"Leaf Node({self.value}, labels={label_counts})"

    def count_correct(self, correct=0, head=True) -> tuple:
        if self.is_leaf():
            label_counts = self.count_labels()
            if not head:
                return label_counts[self.value]
            else:
                return label_counts[self.value], len(self.labels)
        if self.left:
            correct += self.left.count_correct(head=False)
        if self.right:
            correct += self.right.count_correct(head=False)
        if head:
            return correct, len(self.labels)
        else:
            return correct

    def is_leaf(self) -> bool:
        return not self.left and not self.right

    def to_stl(self, prev_formula="", verbose=False) -> str:
        if self.is_leaf():
            return prev_formula + " --> " + self.value + "\n"
        left_spec = prev_formula + (
            self.formula.spec
        )
        right_spec = prev_formula + (self.formula.negate())
        conj = " and " if verbose else " ∧ "
        if self.left and not self.left.is_leaf():
            left_spec = f"({left_spec}){conj}"
        if self.right and not self.right.is_leaf():
            right_spec = f"({right_spec}){conj}"
        left = (
            self.left.to_stl(prev_formula=left_spec, verbose=verbose)
            if self.left
            else ""
        )
        right = (
            self.right.to_stl(prev_formula=right_spec, verbose=verbose)
            if self.right
            else ""
        )
        return left + right

    def classify(self, trace) -> str:
        return self.get_leaf(trace).value
    
    def get_leaf(self, trace):
        if self.value:
            return self
        evaluation = self.formula.evaluate(trace.reshape(1, -1), labels=False)[0]
        rob = evaluation.min()
        next_node = self.left if rob > 0 else self.right
        return next_node.get_leaf(trace)

    def print_tree(self, stem=""):
        print(self)
        if self.left:
            l_stem = stem + "++++"
            print(l_stem, end="")
            self.left.print_tree(stem=l_stem)
        if self.right:
            r_stem = stem + "----"
            print(r_stem, end="")
            self.right.print_tree(stem=r_stem)

    @staticmethod
    def build_tree(traces: np.ndarray, batch_size: int, operators: list, depth=0, max_depth=tree_config["MAX_DEPTH"], binary=False):
        if 0 in traces.shape or traces.ndim == 0:
            return None
        labels = traces[:, -1]
        if len(np.unique(labels)) == 1 or depth == max_depth:
            return TreeNode(None, None, traces, None, value=choose_majority(labels), max_depth=max_depth)
        formula = choose_formula(traces, binary=binary, batch_size=batch_size, operators=operators)
        left_traces, right_traces = split_with_formula(
            traces, formula, return_traces=True, binary=binary
        )
        if len(left_traces) == 0 or len(right_traces) == 0:
            return TreeNode(None, None, traces, None, value=choose_majority(labels), max_depth=max_depth)
        left_node = TreeNode.build_tree(left_traces, batch_size, depth=depth+1, max_depth=max_depth, binary=binary, operators=operators)
        right_node = TreeNode.build_tree(right_traces, batch_size, depth=depth+1, max_depth=max_depth, binary=binary, operators=operators)
        return TreeNode(left_node, right_node, traces, formula, max_depth=max_depth)

    def update_tree(self, batch_size: int, operators: list, trace: np.ndarray, depth=0, binary=False) -> None:
        def rebuild_tree():
            print("Rebuilding tree...")
            new_tree = self.build_tree(self.traces, batch_size=batch_size, depth=depth, binary=binary, max_depth=self.max_depth, operators=operators)
            if new_tree.left or new_tree.right:
                new_tree.value = None
            self.__dict__.update(new_tree.__dict__)
        self.traces = np.vstack((self.traces, trace))
        if self.value:
            if self.value != trace[-1]:
                rebuild_tree()
            return
        rob = self.formula.evaluate(trace.reshape(1, -1), labels=True).min()
        next_node = self.left if rob > 0 else self.right
        expected_label = next_node.value if next_node.is_leaf() else choose_majority(next_node.labels)
        if trace[-1] != expected_label:
            rebuild_tree()
        else:
            next_node.update_tree(trace=trace, batch_size=batch_size, depth=depth+1, binary=binary, operators=operators)