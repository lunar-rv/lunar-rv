print("Loading UI features...")
import numpy as np
import sys
import json
import numpy as np
from preproc import preprocess_trace
from datetime import datetime, timedelta

with open('config.json', 'r') as file:
    config = json.load(file)

def read_user_input(prompt=">") -> str:
    response = input(prompt)
    if not response:
        return "c"
    if response.lower() in "apqwg":
        return response.lower()
    return read_user_input(prompt=prompt)

def print_score(tree):
    print("=" * 50)
    print("STL FORMULA WAS:")
    print(tree.to_stl())
    correct, total = tree.count_correct(head=True)
    print(f"Accuracy: {correct}/{total}")

def print_trees(typed_anom_classifiers, parser):
    def display(sensor_index, sensor_type):
        tree = typed_anom_classifiers[sensor_type][sensor_index]
        if tree is None:
            print("No anomalies detected yet for this sensor.")
        else:
            tree.print_tree()

    while True:
        try:
            index_response = input("Enter sensor index: ")
            sensor_index = int(index_response) - 1
        except ValueError:
            print(f"Invalid sensor index: '{index_response}'")
            continue
        sensor_type = input(f"Enter sensor type ({'/'.join(parser.type)}): ")
        while sensor_type not in parser.type:
            sensor_type = input(f"Invalid sensor type: '{sensor_type}'")
        if not display(sensor_index=sensor_index, sensor_type=sensor_type):
            return
        continue_response = input("Show more graphs? (Y/n): ").lower()
        if continue_response not in ['yes', 'y']:
            return

def progress_bar(index, warmup_stage: int, warmup_times: list, bar_length=40):
    total_time = warmup_times[warmup_stage-1]
    progress = float(index) / float(total_time)
    block = int(bar_length * progress)
    text = (
        f"\rWarm-up {warmup_stage}: Press Enter to continue... "
        f"[{index}/{total_time}] "
        f"[{'#' * block + '-' * (bar_length - block)}]"
    )
    sys.stdout.write(text)
    sys.stdout.flush()

def print_anomaly_info(model, new_batch, formula):
    print("\nAnomaly detected!\n")
    data = preprocess_trace(new_batch=new_batch)
    X = np.delete(data, model.sensor_index, axis=1)
    y = data[:, model.sensor_index].astype(float)
    predictions = model.predict(X)
    print(f"Sensor {model.sensor_index+1} was calculated as:")
    for i, (weight, index) in enumerate(zip(model.coef_, model.indices_used)):
        start = "\t" if i == 0 else " \t+ "
        print(f"{start}Sensor {index+1} x {weight}")
    print(f"Predicted average was {predictions.mean()}")
    print(f"Actual average was {y.mean()}")
    print(f"STL formula was: {formula}")

def get_time_period(new_batch):
    first = new_batch[0] # ... ,284.969993591309,283.569993972778,01/01/2023,00:00:00
    second = new_batch[1] # ...,284.939993858337,283.589993476868,01/01/2023,00:15:00
    time_format = "%H:%M:%S"
    first_time = datetime.strptime(first[-9:-1], time_format)
    second_time = datetime.strptime(second[-9:-1], time_format)
    diff = second_time - first_time
    time_period = int(diff.total_seconds() / 60)
    return time_period

def get_and_display_anomaly_times(anomaly_indices: list, formula, new_batch: list, prev_backlog_size: int, end: int) -> None:
    time_period = get_time_period(new_batch)
    def get_anomaly_bounds(indices) -> list:
        if formula.name == "G":
            return ((0, len(new_batch)),)
        bounds = []
        start_bound = None
        N = len(indices)
        for i, this_value in enumerate(indices):
            if i == 0 or indices[i-1] + 1 != this_value:
                start_bound = this_value
            if i+1 == N or indices[i+1] - 1 != this_value:
                bounds.append((start_bound, this_value + end - 1))
            print(f"Start: {start_bound}, TV: {this_value}, PBS: {prev_backlog_size}, E: {end} I: {indices}")
        return bounds
    print("Violated subformula was:", formula)
    print(f"This means: {formula.human_readable(time_period)}.")
    print("The anomaly was detected between the following times:")
    first_reading_values = new_batch[0].split(",")
    date = first_reading_values[-2]
    time = first_reading_values[-1].strip()
    datetime_str = f"{date} {time}"
    start_time = datetime.strptime(datetime_str, "%d/%m/%Y %H:%M:%S")
    bounds = get_anomaly_bounds(anomaly_indices)
    print("BOUNDS", bounds)
    for interval in bounds:
        interval_start = (start_time + timedelta(minutes = (int(interval[0])) * time_period)).strftime("%d/%m/%Y %H:%M:%S")
        interval_end = (start_time + timedelta(minutes = (int(interval[1])) * time_period)).strftime("%d/%m/%Y %H:%M:%S")
        print(f"\t{interval_start} to {interval_end}")
    return np.array(bounds), start_time
    

def print_intro(types: list, warmup_times: tuple):
    formatted_types = f"{', '.join(types[:-1])} and {types[-1]}"
    print("=" * 65)
    print(f"Online {formatted_types} monitor with LUNAR".center(65))
    print("=" * 65)
    print("Instructions:")
    print("  - Enter : Read the next batch.")
    print("  - 'q'   : Quit the monitor.")
    print("  - 'g'   : Display a graph showing connections between sensors.")
    print("  - 'w'   : Display the weights of the prediction model for a particular sensor.")
    print("  - 'p'   : Display the STL anomaly detection formula for a sensor.")
    print("=" * 65)
    print("\nNote:")
    print(f"  - There are two 'warm-up' phases of length {' and '.join(warmup_times)},")
    print("    which must be completed before monitoring begins.")
    print("=" * 65) 
