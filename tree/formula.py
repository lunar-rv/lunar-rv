from abc import ABC, abstractmethod
import numpy as np
import json
with open("config.json") as config_file:
    config = json.load(config_file)

class Formula:
    def __init__(self, boundary: float, sign=">=", end=None):
        self.boundary = boundary
        self.sign = sign
        self.end = end

    @property
    def stl_spec(self):
        spec = self.spec
        spec = spec.replace("eventually", "F")
        spec = spec.replace("always", "G")
        spec = spec.replace(" s", "")
        return spec
    
    def __repr__(self):
        return self.spec

    @staticmethod
    def build_formula(boundary: float, operator="_", end: int = -1, sign="<="):
        if operator == "_":
            return Basic(boundary, end=end, sign=sign)
        elif operator == "F":
            return Eventually(boundary, end=end, sign=sign)
        elif operator == "G":
            return Always(boundary, end=end, sign=sign)
        else:
            raise ValueError(f"Unknown operator: {operator}")

    @staticmethod
    def list_options(operators="FG_", boundaries=None, binary=True, invariance=False, use_mean=True):
        if invariance:
            if use_mean:
                return [(b, 1, "G") for b in boundaries]
            else:
                return [(b, config["BATCH_SIZE"], "G") for b in boundaries]
        all_options = []
        time_period = config["BATCH_SIZE"]
        def add_options(end, op, b=None):
            if binary:
                all_options.append((0, end, op))
            else:
                all_options.append((b, end, op))
        for op in operators:
            if op in "FG":
                end = 1
                while time_period % end == 0 and time_period > end:
                    if binary:
                        add_options(end, op)
                    else:
                        for b in boundaries:
                            add_options(end, op, b)
                    end *= 2
            else:
                if binary:
                    add_options(1, op)
                else:
                    for b in boundaries:
                        add_options(1, op, b)
        return all_options

    @abstractmethod
    def spec(self):
        pass


    def negate(self, verbose=True):
        if verbose:
            return "not " + self.spec
        else:
            return "¬" + self.stl_spec
    @abstractmethod
    def evaluate(self, trace, labels=True):
        pass


class Basic(Formula):
    def __init__(self, boundary, sign=">=", end=None):
        super().__init__(boundary, sign, end)

    @property
    def spec(self):
        return f"error {self.sign} {self.boundary}"

    def negate(self, verbose=True):
        negated_sign = ""
        if self.sign == ">=":
            negated_sign = "<"
        elif self.sign == "<=":
            negated_sign = ">"
        elif self.sign == ">":
            negated_sign = "<="
        elif self.sign == "<":
            negated_sign = ">="
        return f"error {negated_sign} {self.boundary}"
    
    def evaluate(self, traces, labels=True):
        traces = traces[:, :-1].astype(float) if labels else traces
        return np.mean(self.boundary - traces, axis=1)


class BoundedFormula(Formula, ABC):
    def __init__(self, boundary, sign=">=", end=None):
        super().__init__(boundary, sign, end)
    @abstractmethod
    def evaluate_interval(self, traces):
        pass
    def evaluate(self, traces, labels=True):
        traces = traces[:, :-1].astype(float) if labels else traces
        values = self.evaluate_interval(traces)
        for i in range(1, self.end):
            j = self.end - i
            cut_traces = traces[:, i:-j]
            values = np.hstack((values, self.evaluate_interval(cut_traces)))
        return values.mean(axis=1)

class Eventually(BoundedFormula):
    def __init__(self, boundary, end: int, sign=">="):
        super().__init__(boundary, sign, end)

    @property
    def spec(self):
        return f"eventually[0:{self.end}](error {self.sign} {self.boundary})"
    
    def evaluate_interval(self, traces):
        split_traces = traces.reshape(traces.shape[0], -1, self.end)
        values = self.boundary - np.min(split_traces, axis=2)
        return values

class Always(BoundedFormula):
    def __init__(self, boundary, end: int, sign=">="):
        super().__init__(boundary, sign=sign, end=end)

    @property
    def spec(self):
        if self.end == 1 and config["USE_MEAN"]:
            return f"mean error {self.sign} {self.boundary}"
        if self.end == config["BATCH_SIZE"] and not config["USE_MEAN"]:
            return f"always (error {self.sign} {self.boundary})"
        return f"always[0:{self.end} s](error {self.sign} {self.boundary})"
    
    def evaluate_interval(self, traces):
        split_traces = traces.reshape(traces.shape[0], -1, self.end)
        values = self.boundary - np.max(split_traces, axis=2)
        return values
    
def main():
    residuals = [0.0036163865133644058, 0.04686857007046852, 0.1652082850037706, 0.027735486037190615, 
                 0.12886973531023652, 0.05321521805461485, 0.013912560479747288, 0.004184763925652729,
                 0.05424943527411116, 0.2549802597200182, 0.5577038127065858, 0.23584681074652336, 
                 0.04774402072724171, 0.08651318639462877, 0.030004879181894095, 0.1247800843416011, 
                 0.29869655143784435, 0.19481327480304345, 0.3843290294643087, 0.27134670575677117, 
                 0.17749753198040286, 0.026578510620035162, 0.07932966829624424, 0.04995492033369256, 
                 0.11335375637889361, 0.5303981340045103, 0.6718895053551381, 0.5877984809249601, 
                 0.5419727766902485, 0.06203762211260885, 0.20900903823318703, 0.04290417313913483, 
                 0.04666730660178239, 0.18142692940989122, 0.00362427052912756, 0.12402658248944487, 
                 0.34734412408765625, 0.5207624405086826, 0.4461320381441737, 0.44422900955495837, 
                 0.48059123393830866, 0.5026503523616956, 0.38570928600412807, 0.46438381935492296, 
                 0.58132400495646, 0.1171774580152711, 0.17363025430515086, 0.1171774580152711, 
                 0.022457758649549464, 0.5080197103476715, 0.36085458034024187, 0.4123528304204489, 
                 0.5212494796007285, 0.24664924352357787, 0.0763205384082255, 0.208382345576609, 
                 0.4169771045507767, 0.20812137116076393, 0.1954027664739129, 0.20812137116076393, 
                 0.1825713492733909, 0.1974867632618081, 0.07662852541491169, 0.21662021223530295, 
                 0.5290030122015489, 0.013034387193949087, 0.1508795813294668, 0.0060990617795249336, 
                 0.1386807886589944, 0.4918997824476141, 0.22026851056551958, 0.4727663334741401, 
                 0.8783292642371259, 0.278149748238362, 0.45818432470042314, 0.2398828502914105, 0.0407157487929688, 
                 0.013152671225005047, 0.29429063815539225, 0.013152671225005047, 0.33972720289053226, 
                 0.3357812401771181, 0.3222495401138358, 0.3549146891505921, 0.40671350393453254, 
                 0.050087557605774796, 0.145045215859535, 0.011820659658798999, 0.08313715189047291, 
                 0.032597077967620736, 0.0936664584066417, 0.05173052694111557, 0.047607250672342116, 
                 0.025907592073996094, 0.01895869060497546, 0.0641741250807723
    ]
    import numpy as np
    print(np.mean(residuals))
    print(np.mean(residuals) - 0.1527388)

if __name__ == "__main__":

    main()
