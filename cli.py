import argparse

class CLI:
    def __init__(self):
        """
        This class defines the cli used for our experiments.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--experiment_file', '-e', help='file with model outputs', default = "")
        parser.add_argument('--split', '-s', help='data split', default = "mini")
        parser.add_argument('--device', '-d', help='device', default = "cpu")
        parser.add_argument('--framework', '-f', help='single or multi-agent framework', default = "single")

        self.experiment_file = parser.parse_args().experiment_file
        self.split = parser.parse_args().split
        self.device = parser.parse_args().device
        self.framework = parser.parse_args().framework
