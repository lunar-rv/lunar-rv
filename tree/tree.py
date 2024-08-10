print("Loading code for decision tree construction...")
import numpy as np
from tree.formula import Formula
from tree.compare import compare
import random
from ui import print_score # for testing purposes only
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

    H = l * H_left + r * H_right
    return H

def choose_majority(labels) -> str:
    values, counts = np.unique(labels, return_counts=True)
    choice = values[np.argmax(counts)]
    return choice

def find_best_binary_threshold(values, labels, n=20) -> float:
    safe_values = values[labels == "Safe"]
    anomaly_values = values[labels == "Anomaly"]
    safe_midpoint = np.median(safe_values)
    anomaly_midpoint = np.median(anomaly_values)
    possible_thresholds = np.linspace(anomaly_midpoint, safe_midpoint, n)
    best_threshold = None
    best_accuracy = 0
    for threshold in possible_thresholds:
        classified_labels = np.where(values > threshold, "Anomaly", "Safe")
        accuracy = np.mean(classified_labels == labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    return best_threshold


def split_with_formula(traces: np.ndarray, formula: Formula, return_traces=False, binary=True) -> tuple:
    evaluations = formula.evaluate3(traces).min(axis=1)
    if return_traces:
        left_trace = traces[evaluations > 0]
        right_trace = traces[evaluations <= 0]
        return left_trace, right_trace
    labels = traces[:, -1]
    if binary:
        threshold = find_best_binary_threshold(evaluations, labels, n=20)
        formula.boundary = -threshold
        evaluations = formula.evaluate3(traces).min(axis=1)
    left_lab = labels[evaluations > 0]
    left_rob = evaluations[evaluations > 0]
    right_lab = labels[evaluations <= 0]
    right_rob = evaluations[evaluations <= 0]
    return left_lab, left_rob, right_lab, right_rob


def choose_formula(traces: np.ndarray, binary=False) -> Formula:
    best_entropy = np.inf
    best_formula = None
    values = traces[:, :-1].astype(float)
    boundaries = np.linspace(values.min(), values.max(), num=3)
    for boundary, end, operator in Formula.list_options(binary=binary, boundaries=boundaries):
        formula = Formula.build_formula(boundary=boundary, end=end, operator=operator)
        left_lab, left_rob, right_lab, right_rob = split_with_formula(traces, formula, binary=binary)
        H_1 = stl_entropy(left_lab, left_rob, right_lab, right_rob)
        H_2 = entropy(left_lab, right_lab)
        beta = tree_config["BETA"]
        epsilon = 1e-6
        F_beta = (1 + beta ** 2) * H_1 * H_2 / ((beta ** 2 * H_1 + H_2) + epsilon)
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
            self.formula.spec if verbose else self.formula.stl_spec
        )
        right_spec = prev_formula + (self.formula.negate(verbose=verbose))
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
        if self.value:
            return self.value
        evaluation = self.formula.evaluate3(trace.reshape(1, -1), labels=False)[0]
        rob = evaluation.mean() if config["USE_MEAN"] else evaluation.min()
        next_node = self.left if rob > 0 else self.right
        return next_node.classify(trace)

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
    def build_tree(traces: np.ndarray, depth=0, max_depth=tree_config["MAX_DEPTH"], binary=False):
        if 0 in traces.shape or traces.ndim == 0:
            return None
        labels = traces[:, -1]
        if len(np.unique(labels)) == 1 or depth == max_depth:
            return TreeNode(None, None, traces, None, value=choose_majority(labels), max_depth=max_depth)
        formula = choose_formula(traces, binary=binary)
        left_traces, right_traces = split_with_formula(
            traces, formula, return_traces=True, binary=binary
        )
        if len(left_traces) == 0 or len(right_traces) == 0:
            return TreeNode(None, None, traces, None, value=choose_majority(labels), max_depth=max_depth)
        left_node = TreeNode.build_tree(left_traces, depth=depth+1, max_depth=max_depth, binary=binary)
        right_node = TreeNode.build_tree(right_traces, depth=depth+1, max_depth=max_depth, binary=binary)
        return TreeNode(left_node, right_node, traces, formula, max_depth=max_depth)

    def update_tree(self, trace: np.ndarray, depth=0, binary=False) -> None:
        def rebuild_tree():
            print("Rebuilding tree...")
            new_tree = self.build_tree(self.traces, depth=depth, binary=binary, max_depth=self.max_depth)
            if new_tree.left or new_tree.right:
                new_tree.value = None
            self.__dict__.update(new_tree.__dict__)
        self.traces = np.vstack((self.traces, trace))
        if self.value:
            if self.value != trace[-1]:
                rebuild_tree()
            return
        rob = self.formula.evaluate(trace.reshape(1, -1), labels=True)
        next_node = self.left if rob > 0 else self.right
        expected_label = next_node.value if next_node.is_leaf() else choose_majority(next_node.labels)
        if trace[-1] != expected_label:
            rebuild_tree()
        else:
            next_node.update_tree(trace, depth=depth+1, binary=binary)
                

def small():
    return random.gauss(2, 1)


def medium():
    return random.gauss(3, 1)


def large():
    return random.gauss(4, 1)


def main():
    UPDATE = True
    random.seed(42)
    num_examples = 5
    false_alarms = np.array(
        [
            [medium(), small(), small(), small(), large(), small(), medium(), small()] + ["false alarm"]
            for _ in range(num_examples)
        ]
    )
    # step_faults = np.array([[medium(), medium(), small(), large(), medium()] +["step fault"] for _ in range(num_examples)])
    # ramp_faults = np.array([[large(), small(), large(), large(), medium()] + ["ramp fault"] for _ in range(num_examples)])
    # traces = np.vstack((false_alarms, step_faults, ramp_faults))
    # tree = Node.build_tree(traces)
    # tree.print_tree()
    # print_score(tree)
    traces = false_alarms

    if UPDATE:
        tree = TreeNode.build_tree(traces, max_depth=tree_config["MAX_DEPTH"])
        tree.print_tree()
        num_updates = 3
        new_false_alarms = np.array(
            [
                [medium(), small(), small(), small(), large(), small(), medium(), small()] + ["false alarm"]
                for _ in range(num_updates)
            ]
        )
        new_step_faults = np.array(
            [
                [medium(), medium(), small(), large(), medium(), medium(), small(), medium()] + ["step fault"]
                for _ in range(num_updates)
            ]
        )
        new_ramp_faults = np.array(
            [
                [large(), small(), large(), large(), medium(), large(), large(), medium()] + ["ramp fault"]
                for _ in range(num_updates)
            ]
        )
        new_traces = np.vstack((new_step_faults, new_false_alarms, new_ramp_faults))
        for t in new_traces:
            print(f"Updating with new trace: {t}")
            tree.update_tree(t)
            tree.print_tree()
        print_score(tree)
        traces = np.vstack((traces, new_traces))
    new_trace = np.array([3, 2.5, 2.5, 2.1, 1.9])
    label = tree.classify(new_trace)
    print("New trace is classified as:", label)
    print("=" * 50)
    compare(traces)

def test_formula_choice():
    traces = np.array(
        [
            [
                "2.6966250903907714",
                "4.379278558940257",
                "0.6597879962893545",
                "3.3250793654472925",
                "4.539343562372228",
                "step fault",
            ],
            [
                "2.0121587938482146",
                "5.1129167229278405",
                "1.8542564804072446",
                "4.2621021712229075",
                "4.033231829126633",
                "step fault",
            ],
            [
                "4.217024274483255",
                "3.5211997578859293",
                "0.9738198001929221",
                "3.1588943443926927",
                "4.167249204341795",
                "step fault",
            ],
            [
                "5.2048349333331725",
                "0.4140925536018947",
                "3.2184417587032312",
                "4.249048507200529",
                "3.116865597963296",
                "ramp fault",
            ],
            [
                "5.021137490791534",
                "2.073102287961573",
                "4.183463272528609",
                "1.8249435631146262",
                "3.285640604985273",
                "ramp fault",
            ],
        ]
    )
    formula = choose_formula(traces=traces)
    left_traces, right_traces = split_with_formula(traces, formula, return_traces=True)
    print("Left:", left_traces)
    print("Right:", right_traces)
    print(formula.spec)

def test_entropy():
    traces = np.array([
        [1,1,1,1,'s'],
        [0.5,0.5,0.5,0.5,'s'],
        [5,5,5,5,'l'],
    ])
    formula = choose_formula(traces)
    print(formula)

if __name__ == "__main__":
    test_formula_choice()
