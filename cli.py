import argparse
import json
import os
import pdb

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
        else:
            raise NameError(f"No such file: {config_file}")
        #else:
        #     self.experiment_file = parser.parse_args().experiment_file
        #     self.split = parser.parse_args().split
        #     self.device = parser.parse_args().device
        #     self.framework = parser.parse_args().framework

        # add default configurations
        # if "device" not in self.config.keys():
        #     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # ensure nvidia order
        #     os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set to first GPU in nvidia order (only this one is visible)

        #     import torch
        #     self.config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu" # set to first of visible GPUs
        #     print("Device: ", torch.cuda.get_device_name(torch.cuda.current_device()))
        # else:
        #     if self.config["device"] == "hpc":
        #         import torch
        #         self.config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        #         print("Cuda?", torch.cuda.is_available())
        #     elif self.config["device"] not in ["mig", "hpc"]:
        #         os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # ensure nvidia order
        #         os.environ["CUDA_VISIBLE_DEVICES"] = self.config["device"].split(":")[-1] # set to n GPU in nvidia order (only this one is visible)
        #         print("Visible device:", os.environ["CUDA_VISIBLE_DEVICES"])
        #         import torch
        #         print("GPU: ", torch.cuda.get_device_name(torch.cuda.current_device()))
        #     else:
        #         self.config["device"] = "cuda:0" # # set to first of visible GPUs = visible device

        #         import torch
        #         print("GPU: ", torch.cuda.get_device_name(torch.cuda.current_device()))

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

        # if "interactive" not in self.config.keys():
        #     self.config["interactive"] = False
        # else:
        #     if self.config["interactive"].lower() == "true":
        #         self.config["interactive"] = True
        #     else:
        #         self.config["interactive"] = False
