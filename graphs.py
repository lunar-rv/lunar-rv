import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import logging

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


def plot_array(trace: np.ndarray, sensor_index: int, keyword: str):
    plt.plot(trace, label=f"{keyword}")
    plt.xlabel("Time")
    plt.ylabel(f"Sensor {sensor_index}")
    plt.title(f"{keyword} for Sensor {sensor_index}")
    plt.legend()
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot traces")
    parser.add_argument(
        "-p",
        "--pos_infile",
        default="csv/positive_test.csv",
        help="Positives input file",
    )
    parser.add_argument(
        "-n",
        "--neg_infile",
        default="csv/negative_test.csv",
        help="Negatives input file",
    )
    parser.add_argument(
        "-o", "--outfile", default="images/residuals.png", help="Output file for graph"
    )
    args = parser.parse_args()
    plot_traces(args.neg_infile, args.pos_infile, args.outfile)


if __name__ == "__main__":
    main()
