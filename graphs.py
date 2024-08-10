import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import json

with open("config.json", "r") as config_file:
    config = json.load(config_file)

logging.getLogger('matplotlib').setLevel(logging.WARNING)

def plot_traces(neg_infile, pos_infile, outfile):
    neg_df = pd.read_csv(neg_infile)
    pos_df = pd.read_csv(pos_infile)
    negatives = neg_df.to_numpy().flatten()
    positives = pos_df.to_numpy().flatten()
    plt.plot(negatives[:96], label="Negatives", color="blue")
    plt.plot(positives[:96], label="Positives", color="red")
    plt.xlabel("Time")
    plt.ylabel("Magnitude of residual for day 1")
    plt.legend()
    plt.savefig(outfile)
    plt.close()


def plot_array(trace: np.ndarray, sensor_index: int, batch_start_time: datetime, keyword: str, sensor_type: str, backlog_size: int = 0, boundary=None, bounds: list = []):
    time_period = config["TIME_PERIOD"]
    trace_start_time = batch_start_time - timedelta(minutes=backlog_size * time_period)
    trace_end_time = trace_start_time + timedelta(minutes=(len(trace) - 1) * time_period)
    int_ticks = np.linspace(0, len(trace), 9)
    dt_ticks = [trace_start_time + timedelta(minutes=int(tick) * time_period) for tick in int_ticks]
    plt.plot(trace, label=f"{keyword}", color='blue')
    for start, end in bounds:
        plt.axvspan(start, end, color='orange', alpha=0.3, label="Anomaly" if start == bounds[0][0] else "")
    plt.xlabel("Time")
    plt.xticks(int_ticks, [dt.strftime("%H:%M") for dt in dt_ticks])
    units = "mBar" if sensor_type == "PRESSURE" else "Kelvin"
    plt.ylabel(f"Sensor {sensor_index+1} {keyword} ({units})")
    if boundary is not None:
        plt.axhline(y=boundary, color="red", linestyle="--", label="Formula boundary")
    if trace_start_time == trace_end_time:
        date_str = f"on {trace_start_time.strftime('%Y/%m/%d')}"
    else:
        date_str = f"from {trace_start_time.strftime('%Y/%m/%d')} to {trace_end_time.strftime('%Y/%m/%d')}"
    plt.title(f"{keyword} for {sensor_type.capitalize()} Sensor {sensor_index+1} {date_str}")
    if boundary is not None:
        rob_metric = "mean" if config["USE_MEAN"] else "min"
        plt.text(0.95, 0.01, f"Robustness metric: {rob_metric}", transform=plt.gca().transAxes,
         fontsize=8, verticalalignment='bottom', horizontalalignment='right')
    plt.legend()
    plt.show()
