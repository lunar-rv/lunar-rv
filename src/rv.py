from parser import Parser
from monitor import run_monitor
import argparse

print("Parsing specification...")

argparser = argparse.ArgumentParser()
argparser.add_argument("spec_file", help="The file containing the specification")
args = argparser.parse_args()

def main():
    p = Parser()
    p.parse(args.spec_file)
    run_monitor(p)

if __name__ == "__main__":
    main()
