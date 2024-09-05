from parser import Parser
from monitor import run_monitor
import argparse
from offline import *
import json
with open("config.json", "r") as config_file:
    config = json.load(config_file)

print("Parsing specification...")

argparser = argparse.ArgumentParser()
argparser.add_argument("spec_file", help="The file containing the specification")
argparser.add_argument("--test", help="The test to run")
args = argparser.parse_args()

def main():

    p = Parser()
    p.parse(args.spec_file)
    if config["ONLINE"]:
        run_monitor(p)
    else:
        import sys
        if len(sys.argv) == 3:
            func = globals()[f"testing_{args.test}"]
            func(p)
        else:
            testing_6(p)


if __name__ == "__main__":
    main()
