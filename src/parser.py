import re

class Parser:
    def __init__(self):
        self.inputs = set(["stl", "type", "batch", "safe"])
        self.stl = []
        self.type = []
        self.type_indices = [0]
        self.patterns = {
            "input": r'^input\s*=\s*"?(?P<var>.+?)"?\s*$',
            "stl":  r'^add stl\s*(?P<var>F|G|G_avg)$',      # Matches F, G, or G_avg
            "type": r'^add type\s+(?P<var>\w+)\s+(?P<sensors>\d+)$',  # Matches 1) any non-empty string 2) any int
            "batch": r'^batch\s*=\s*(?P<var>[1-9]\d*)$',        # Matches any positive integer
            "safe": r'^safe\s*=\s*(?P<var>[1-9]\d*)$',          # Matches any positive integer
        }
        self.human_readable = {
            "input": 'input = "filename.csv"  # Any valid file name',
            "stl": 'add stl F | G | G_avg  # One of "F", "G", or "G_avg"',
            "type": 'add type "type_name"  # Any non-empty string',
            "batch": 'batch = positive_integer  # Any positive integer, e.g., 1, 2, 100',
            "safe": 'safe = positive_integer  # Any positive integer, e.g., 1, 2, 100',
        }
    def parse_input(self, input_line):
        match = re.match(self.patterns["input"], input_line)
        if not match:
            raise ValueError(f"Invalid input line: {input_line}")
        self.infile = match.group("var")

    def get_prefix(self, line):
        words = line.split(" ")
        return words[1] if words[0] == "add" else words[0]
    
    def parse_line(self, line):
        prefix = self.get_prefix(line)
        pattern = self.patterns.get(prefix)
        match = re.match(pattern, line)
        if not match:
            raise ValueError(f"Invalid expression for {prefix}: {line}\n"
                            f"Expected format: {self.human_readable[prefix]}")
        else:
            variable = match.group("var")
            if line.startswith("add"):
                self.__dict__[prefix].append(variable)
                if prefix == "type":
                    prev = self.type_indices[-1]
                    num_sensors = match.group("sensors")
                    self.type_indices.append(prev + int(num_sensors))
            else:
                self.__dict__[prefix] = int(variable)

    def parse(self, spec_file):
        with open(spec_file) as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip() and line[0] != "#"]
        input_line = lines[0]
        self.parse_input(input_line)
        line_starts = [self.get_prefix(line) for line in lines[1:]]
        line_starts = set(sorted(line_starts))
        unrecognised = line_starts - self.inputs
        missing = self.inputs - line_starts
        if unrecognised:
            raise ValueError(f"Unrecognised inputs: {unrecognised}")
        if missing:
            raise ValueError(f"Missing inputs: {missing}")
        for line in lines[1:]:
            self.parse_line(line)
        if len(self.stl) != len(set(self.stl)):
            raise ValueError(f"Duplicate STL operators: {self.stl}")

def main():
    p = Parser()
    p.parse("spec.file")
    print(p.__dict__)
if __name__ == "__main__":
    main()