import argparse
import json
import os

class CLI:
    def __init__(self):
        """
        This class defines the cli used for our experiments.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', '-c', help='configuration file', default = "configs/mini_cpu_single_exp.json")

        config_file = parser.parse_args().config
        if os.path.isfile(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            raise NameError(f"No such file: {config_file}")

        if self.config["framework"] == "multi" and "conversation_rounds" not in self.config.keys():
            self.config["conversation_rounds"] = 2

        # prompt candidate
        if "candidate" not in self.config.keys():
            self.config["candidate"] = "adapted-agashe-2025_str_gemini-final"

        if "variation" not in self.config.keys():
            self.config["variation"] = False
        else:
            if self.config["variation"].lower() == "true":
                self.config["variation"] = True
            else:
                self.config["variation"] = False

        if "final_reorder" not in self.config.keys():
            self.config["final_reorder"] = False
        else:
            if self.config["final_reorder"].lower() == "true":
                self.config["final_reorder"] = True
            else:
                self.config["final_reorder"] = False

        if "structured" not in self.config.keys():
            self.config["structured"] = False
        else:
            if self.config["structured"].lower() == "true":
                self.config["structured"] = True
            else:
                self.config["structured"] = False

        if "final" not in self.config.keys():
            self.config["final"] = True
        else:
            if self.config["final"].lower() == "true":
                self.config["final"] = True
            else:
                self.config["final"] = False

        if "start" not in self.config.keys():
            self.config["start"] = 0

        if "end" not in self.config.keys():
            self.config["end"] = None
