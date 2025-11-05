import argparse

class CLI:
    def __init__(self):
        """
        This class defines the cli used for our experiments.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--experiment_file', '-e', help='file with model outputs', default = "./outputs/single/Llama-3.2-1B-Instruct/outputs_single_Llama-3.2-1B-Instruct_2025-11-05_16:39:01.188709.jsonl")

        self.experiment_file = parser.parse_args().experiment_file
