import argparse
import json
import os

class CLI:
    def __init__(self):
        """
        This class defines the cli used for our experiments.
        """
        parser = argparse.ArgumentParser()
        # parser.add_argument('--experiment_file', '-e', help='file with model outputs', default = "")
        # parser.add_argument('--split', '-s', help='data split', default = "mini")
        # parser.add_argument('--device', '-d', help='device', default = "cpu")
        # parser.add_argument('--framework', '-f', help='single or multi-agent framework', default = "single")
        parser.add_argument('--config', '-c', help='configuration file', default = "configs/mini_cpu_single_exp.json")

        config_file = parser.parse_args().config
        if os.path.isfile(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        # else:
        #     self.experiment_file = parser.parse_args().experiment_file
        #     self.split = parser.parse_args().split
        #     self.device = parser.parse_args().device
        #     self.framework = parser.parse_args().framework
