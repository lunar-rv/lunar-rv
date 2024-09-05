# from offline import apply_anomaly
from preproc import preprocess_trace
import numpy as np
import matplotlib.pyplot as plt

infile = "inputs/traces.csv"
batch_size = 96

def apply_anomaly(dataset, anom_type, uniform_size=1, gauss_size=1, spike_size=3, ramp_size=2):
    if anom_type == "gauss":
        anom = np.random.normal(0, gauss_size * dataset.std(), dataset.shape)
        return dataset + anom
    elif anom_type == "uniform":
        anom = dataset.std() * uniform_size
        return dataset + anom
    elif anom_type == "spike":
        spike_size = dataset.std() * spike_size
        for i in range(dataset.shape[0]):
            spike_index = np.random.choice(dataset.shape[1])
            dataset[i, spike_index] += spike_size
        return dataset
    elif anom_type == "ramp":
        N = dataset.shape[1]
        ramp = ramp_size * dataset.std() * ((np.arange(N) + 1) / N)
        # print(ramp)
        # print(dataset.std())
        return dataset + ramp
    elif anom_type == "hang":
        dataset[0, 60:] = dataset[0, 60]
        return dataset
    else:
        raise ValueError(f"Invalid anomaly type: '{anom_type}'")
        

data = preprocess_trace(infile=infile)
first = data[0:batch_size, 0].reshape(1, -1)
fc = first.copy().flatten()
noise = apply_anomaly(first, "gauss", gauss_size=1).flatten()
ramp = apply_anomaly(first, "ramp", uniform_size=1).flatten()
spike = apply_anomaly(first, "spike", spike_size=3).flatten()
hang = apply_anomaly(first, "hang").flatten()
fig, axs = plt.subplots(4, 1, figsize=(10, 15))  # 4 rows, 1 column of subplots

# Plot Normal vs Noise fault
axs[0].plot(fc.flatten(), label="Normal", linestyle='-', marker='o', alpha=0.7)
axs[0].plot(noise.flatten(), label="Noise fault", linestyle='--', marker='x', alpha=0.7)
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Sensor Value")
axs[0].legend()
axs[0].set_title("Normal vs Noise Fault")

# Plot Normal vs Drift fault
axs[1].plot(fc.flatten(), label="Normal", linestyle='-', marker='o', alpha=0.7)
axs[1].plot(ramp.flatten(), label="Drift fault", linestyle='-.', marker='s', alpha=0.7)
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Sensor Value")
axs[1].legend()
axs[1].set_title("Normal vs Drift Fault")

# Plot Normal vs Spike fault
axs[2].plot(fc.flatten(), label="Normal", linestyle='-', marker='o', alpha=0.7)
axs[2].plot(spike.flatten(), label="Spike fault", linestyle=':', marker='d', alpha=0.7)
axs[2].set_xlabel("Time")
axs[2].set_ylabel("Sensor Value")
axs[2].legend()
axs[2].set_title("Normal vs Spike Fault")

# Plot Normal vs Hang fault
axs[3].plot(fc.flatten(), label="Normal", linestyle='-', marker='o', alpha=0.7)
axs[3].plot(hang.flatten(), label="Hang fault", linestyle='-', marker='^', alpha=0.7)
axs[3].set_xlabel("Time")
axs[3].set_ylabel("Sensor Value")
axs[3].legend()
axs[3].set_title("Normal vs Hang Fault")

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()