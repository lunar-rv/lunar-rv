from sklearn.tree import DecisionTreeClassifier
import numpy as np


def compare(traces):
    values = np.delete(traces, -1, axis=1)
    labels = traces[:, -1]
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(values, labels)
    predictions = model.predict(values)
    correct = np.where(predictions == labels, 1, 0)
    print(f"Accuracy of other model: {correct.sum()}/{len(labels)}")


def main():
    def small():
        return random.gauss(2, 1)

    def medium():
        return random.gauss(4, 1)

    def large():
        return random.gauss(6, 1)

    import random

    random.seed(42)
    num_examples = 5
    false_alarms = np.array(
        [
            [medium(), small(), small(), small(), large()] + ["false alarm"]
            for _ in range(num_examples)
        ]
    )
    step_faults = np.array(
        [
            [medium(), medium(), small(), large(), medium()] + ["step fault"]
            for _ in range(num_examples)
        ]
    )
    ramp_faults = np.array(
        [
            [large(), small(), large(), large(), medium()] + ["ramp fault"]
            for _ in range(num_examples)
        ]
    )
    traces = np.vstack((false_alarms, step_faults, ramp_faults))
    compare(traces)


if __name__ == "__main__":
    main()
