

from ui import read_user_input, read_anomaly_type, parse_input_args, print_trees, progress_bar, show_weights, plot_graph
from file_io import clear_files, write_header, get_new_batch, write_new_batch, end_anomaly, start_anomaly
from model import get_residuals, update_spec, log_anomaly, new_batch_ok
import json
with open("config.json", 'r') as config_file:
    config = json.load(config_file)

def monitor_loop(args) -> None:
    index = 0
    warmup1 = True
    warmup2 = False
    safe_trace_file = args.safe_trace_file
    anom_classifier = None
    formula = None
    bin_classifier = None
    progress_bar(index=0, warmup_stage=1)
    last_reading_safe: bool = True
    while True:
        prompt = "" if warmup1 or warmup2 else ">"
        response = read_user_input(prompt)
        if response == "q":
            print("Exiting monitor...")
            exit()
        if response == "p":
            print_trees(bin_classifier=bin_classifier, anom_classifier=anom_classifier)
            continue
        if response == "w":
            show_weights(sensor_index=0)
            continue
        if response == "g":
            plot_graph()
            continue
        anomaly_type = read_anomaly_type() if response == "a" else None
        new_batch = get_new_batch(
            batch_size=args.batch_size,
            num_sensors=args.num_sensors,
            index=index,
            source_file=args.source_file,
        )
        index += 1
        if warmup1:
            progress_bar(warmup_stage=1, index=index)
            if index >= args.warmup_time:
                warmup1 = False
                warmup2 = True
                print("\nWarmup 1 complete.")
            write_new_batch(new_batch=new_batch, outfile=args.safe_trace_file)
            continue
        num_evaluations = 1  ## Just leave this as 1. Trace and spec file separation will need sorting out later
        for sensor_index in range(num_evaluations):
            residuals = get_residuals(
                safe_trace_file=safe_trace_file,
                new_batch=new_batch,
                sensor_index=sensor_index,
                anomaly_type=anomaly_type,
            )
            new_trace = ",".join(residuals.astype(str))
            if not new_batch_ok(residuals=residuals, formula=formula, new_batch=new_batch):
                start_anomaly(new_batch, sensor_index + 1)
                last_reading_safe = False
                grow_tree: bool = not anom_classifier and not warmup2
                anom_classifier = log_anomaly(
                    new_batch, new_trace, sensor_index + 1, anom_classifier, grow_tree
                )
                formula, bin_classifier = update_spec(
                    spec_file=config["SPEC_FILE"],
                    residuals_file=args.residuals_file,
                    anomalies_file=config["ANOMALIES_FILE"],
                    bin_classifier=bin_classifier,
                    new_trace=residuals,
                    new_label="Anomaly",
                )
                continue
            if not last_reading_safe:
                end_anomaly(new_batch, sensor_index + 1)
                last_reading_safe = True
            write_new_batch(new_batch=new_batch, outfile=args.safe_trace_file)
            with open(args.residuals_file, "a") as f:
                f.write("\n" + new_trace)
            if not warmup2:
                formula, bin_classifier = update_spec(
                    spec_file=config["SPEC_FILE"],
                    residuals_file=args.residuals_file,
                    anomalies_file=config["ANOMALIES_FILE"],
                    bin_classifier=bin_classifier,
                    new_trace=residuals,
                    new_label="Safe",
                )
            else:
                progress_bar(warmup_stage=2, index = index - config["WARMUP_TIME"])
                if index >= args.warmup_time * 2:
                    warmup2 = False
                    print("\nWarmup complete.")


def main() -> None:
    args = parse_input_args()
    write_header(args.source_file, args.safe_trace_file)
    clear_files(config["LOG_FILE"], config["SPEC_FILE"], config["RESIDUALS_FILE"], config["ANOMALIES_FILE"])
    print("Welcome to the online sensor monitor using linear regression and STL")
    print("Press Enter to read next batch, q to quit, a to add an anomaly: ")
    monitor_loop(args)

if __name__ == "__main__":
    main()
