from parser import Parser
from monitor import run_monitor
import argparse
from offline import *
import json

# Load configuration file
with open("config.json", "r") as config_file:
    config = json.load(config_file)

print("Parsing specification...")

# Argument parser setup
argparser = argparse.ArgumentParser()
argparser.add_argument("spec_file", help="The file containing the specification")
argparser.add_argument("--test", default=-1, help="The test to run")
argparser.add_argument("--index", nargs='?', default=None, help="Sensor index to use (optional)")
argparser.add_argument("--model", nargs='?', default=None, help="Model to use (optional)")
argparser.add_argument("--offline", "-o", action="store_true", help="Run the monitor online")
args = argparser.parse_args()

def main():

    p = Parser()
    p.parse(args.spec_file)

    if config["ONLINE"] and not args.offline:
        run_monitor(p)
    else:
        if args.test == -1:
            print("Error: no test specified.")
            print("Offline mode is for tool evaluation only.")
            print("In order to run the monitor online, please set the ONLINE flag to True in the config file.")
            print("Otherwise, please specify a test to run, e.g. python src/rv.py example.spec --test 1")
            exit()
        import sys
        # Check if a test function exists
        if len(sys.argv) >= 3:
            func = globals()[f"testing_{args.test}"]
            
            # Handle test 4 with optional index and model arguments
            if int(args.test) == 4:
                func(p, s_index=int(args.index) if args.index else None, model_type=args.model)
            else:
                func(p)
        else:
            # Default to running test 6 if no other test is specified
            testing_6(p)

if __name__ == "__main__":
    main()