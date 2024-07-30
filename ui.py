import argparse
import sys
import json
with open('config.json', 'r') as file:
    config = json.load(file)

def read_user_input(prompt=">") -> str:
    response = input(prompt)
    if not response:
        return "c"
    if response.lower() in "apq":
        return response.lower()
    return read_user_input(prompt=prompt)

def read_anomaly_type() -> str:
    print("Choose anomaly type:")
    print("Press 1 for a single sensor increase")
    print("Press 2 for a big single sensor increase")
    print("Press 3 for a network-wide increase")
    print("Or press 4 to enter a normal batch:", end=" ")
    response = input()
    while response not in "1234":
        response = input()
    return response if response != "4" else None

def parse_input_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online STL monitor")
    parser.add_argument(
        "-s",
        "--source-file",
        type=str,
        default=config["SOURCE_FILE"],
        help="File in which sensor values are stored, from which they are fed into the tool",
    )
    parser.add_argument(
        "-r",
        "--residuals-file",
        type=str,
        default=config["RESIDUALS_FILE"],
        help="File in which the residuals from the sensor predictions are stored",
    )
    parser.add_argument(
        "-a",
        "--anomaly-size",
        type=float,
        default=config["ANOMALY_SIZE"],
        help="The size of the anomaly to insert into the traces",
    )
    parser.add_argument(
        "-N",
        "--num-sensors",
        type=int,
        default=config["NUM_SENSORS"],
        help="Number of sensor reading types in the dataset",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="images/heatmap.png",
        help="Output file for the linear regression heatmap",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=config["BATCH_SIZE"],
        help="Number of sensor readings to parse in one batch",
    )
    parser.add_argument(
        "-i",
        "--safe-trace-file",
        type=str,
        default=config["SAFE_TRACE_FILE"],
        help="File where new sets of readings are stored",
    )

    parser.add_argument(
        "-w",
        "--warmup-time",
        type=int,
        default=config["WARMUP_TIME"],
        help="File where new sets of readings are stored",
    )
    args = parser.parse_args()
    return args

def print_score(tree):
    print("=" * 50)
    print("STL FORMULA WAS:")
    print(tree.to_stl())
    correct, total = tree.count_correct(head=True)
    print(f"Accuracy: {correct}/{total}")


def print_trees(bin_classifier, anom_classifier):
    if bin_classifier:
        print("ANOMALY/SAFE tree:")
        bin_classifier.print_tree()
        print_score(bin_classifier)
    if anom_classifier:
        print("ANOMALY TYPE CLASSIFICATION tree")
        anom_classifier.print_tree()
        print_score(anom_classifier)

def progress_bar(index, warmup_stage: int, bar_length=40):
    progress = float(index) / float(config["WARMUP_TIME"])
    block = int(round(bar_length * progress))
    text = f"\rWarmup {warmup_stage}: Press Enter to continue... [{index}/{config['WARMUP_TIME']}] [{'#' * block + '-' * (bar_length - block)}]"
    sys.stdout.write(text)
    sys.stdout.flush()
