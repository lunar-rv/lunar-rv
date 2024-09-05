import os
import time
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.rcParams['text.usetex'] = False
plt.style.use("default")

def load_indices(model_type):
    indices = np.zeros((200))
    sensor_index = 0
    while True:
        try:
            with open(f"error_indices_{sensor_index}_{model_type}.npy", "rb") as f:
                indices += np.load(f)
            sensor_index += 1
        except FileNotFoundError:
            print(f"File {sensor_index} not found")
            indices *= 0
            sensor_index = 0
            time.sleep(20)
        if sensor_index == 54:
            return indices.astype(int)
    

def plot_indices(old_indices, new_indices):
    plt.title("Evaluation of update_tree function")
    plt.xlabel("Batch number")
    plt.ylabel("Number of classification errors")
    plt.plot(old_indices, label="Partial Rebuilding")
    plt.plot(new_indices, label="Leaf Rebuilding")
    plt.legend()
    plt.show()

def run_experiments(delete_files=True):
    os.system("echo 'Running the script'")
    if delete_files:
        os.system("rm -f *_old.npy")
    os.system("./ac.sh old")
    now = time.time()
    old_indices = load_indices("old")
    print("Time taken to load old indices:", time.time() - now)
    print("OLD:", old_indices.tolist())
    if delete_files:
        os.system("rm -f *_new.npy")
    now = time.time()
    os.system("./ac.sh new")
    new_indices = load_indices("new")
    print("Time taken to load new indices:", time.time() - now)
    print("NEW:", new_indices.tolist())
    return old_indices, new_indices

def main():
    old, new = run_experiments(delete_files=True)
    plot_indices(old, new)
    # new = load_indices("new")
    # plt.plot(new)
    # plt.show()

if __name__ == "__main__":
    main()