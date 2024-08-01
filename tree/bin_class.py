import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tree.tree import TreeNode
from sklearn.model_selection import train_test_split


def build(neg_train, pos_train, invariance=False, use_mean=True):
    neg_label_train = np.full((neg_train.shape[0], 1), "Safe")
    pos_label_train = np.full((pos_train.shape[0], 1), "Anomaly")
    train_data = np.vstack(
        (
            np.hstack((neg_train, neg_label_train)),
            np.hstack((pos_train, pos_label_train)),
        )
    )
    head = TreeNode.build_tree(train_data, max_depth=1, binary=True, invariance=invariance, use_mean=use_mean)
    return head


def update(tree, new_values, label, invariance=False, use_mean=True):
    label = np.array([label])
    new_trace = np.hstack((new_values, label))
    tree.update_tree(new_trace, binary=True, invariance=invariance, use_mean=use_mean)
    return tree


def main():
    # predict(anomaly_size=0.0002, num_sensors=1)
    # Load negative and positive values
    neg_values = np.genfromtxt("csv/negative_val.csv", delimiter=",")
    pos_values = np.genfromtxt("csv/positive_val.csv", delimiter=",")

    # Split the data into training and testing sets
    neg_train, neg_test = train_test_split(neg_values, test_size=0.5, random_state=42)
    pos_train, pos_test = train_test_split(pos_values, test_size=0.5, random_state=42)

    # Combine the training data
    formula = build(neg_train=neg_train, pos_train=pos_train).formula

    # Evaluate the model on the test set
    neg_predictions = formula.evaluate(neg_test)
    pos_predictions = formula.evaluate(pos_test)
    # Combine classifications and ground truths
    print(neg_predictions.shape)
    predictions = np.vstack((neg_predictions, pos_predictions)).flatten()
    print(predictions.shape)
    bool_predictions = np.where(predictions > 0, False, True)
    ground_truth = np.hstack(
        (np.full(neg_test.shape[0], False), np.full(pos_test.shape[0], True))
    )

    print("GT:", ground_truth, "PREDICTIONS:", bool_predictions)
    print(f"Accuracy: {accuracy_score(ground_truth, bool_predictions)}")
    print(
        f"Precision: {precision_score(ground_truth, bool_predictions, zero_division=0)}"
    )
    print(f"Recall: {recall_score(ground_truth, bool_predictions, zero_division=0)}")
    print(f"F1: {f1_score(ground_truth, bool_predictions, zero_division=0)}")


if __name__ == "__main__":
    main()
