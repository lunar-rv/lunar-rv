import numpy as np
from tree.tree import TreeNode


def build(neg_train, pos_train, operators):
    neg_label_train = np.full((neg_train.shape[0], 1), "Safe")
    pos_label_train = np.full((pos_train.shape[0], 1), "Anomaly")
    train_data = np.vstack(
        (
            np.hstack((neg_train, neg_label_train)),
            np.hstack((pos_train, pos_label_train)),
        )
    )
    batch_size = neg_train.shape[1]
    head = TreeNode.build_tree(train_data, batch_size=batch_size, max_depth=3, binary=True, operators=operators)
    # head.print_tree()
    return head


def update(tree, new_values, label, operators):
    label = np.array([label])
    new_trace = np.hstack((new_values, label))
    batch_size = new_values.size
    tree.update_tree(trace=new_trace, batch_size=batch_size, binary=True, operators=operators)
    return tree