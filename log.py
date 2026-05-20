from collections import defaultdict
import os
import json
import csv
import pandas as pd
import re
import wandb
import torch

class ExperimentLog:
    def __init__(self, config: dict = None):
        """
        This class logs the important information for an experiment run.

        @params
            config: configuration data that should be logged
        """
        wandb.login()
        self.run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="j-grimm-ludwig-maximilian-university-of-munich",
            # Set the wandb project where this run will be logged.
            project="CollaborationLearning",
            # Track hyperparameters and run metadata.
            config=config
        )

        # self.output_artifact = wandb.Artifact(name="output", type="dataset")
        
        # init logger
        self.run_data = defaultdict()

        # log config data
        if config:
            self.run_data.update(config)
            self.run.config.update(config)

            self.run.config.update({"GPU": torch.cuda.get_device_name(torch.cuda.current_device())})

            # Assign and print run_idx
            self.store_run_data()
            print(f"Batch size: {self.run_data['batch_size']}")
                

    def log_single(self, single):
        """
        This method loggs all attributes of an Single class object

        @params
            single: object of the class Single
        """
        self.run_data.update({"user_prompt": single.PROMPT.replace("\n", "\\n")})
        self.run.config.update({"user_prompt": single.PROMPT.replace("\n", "\\n")})

    def log_multi(self, multi):
        """
        This method loggs all attributes of an Multi class object

        @params
            multi: object of the class Multi
        """
        self.run_data.update({"system_prompt": multi.SYSTEM_PROMPT.replace("\n", "\\n")})
        self.run_data.update({"start_prompt": multi.START_PROMPT.replace("\n", "\\n")})
        self.run_data.update({"mid_prompt": multi.MID_PROMPT.replace("\n", "\\n")})
        self.run_data.update({"rules": multi.RULES.replace("\n", "\\n")})
        self.run_data.update({"names": multi.NAMES})
        self.run_data.update({"conversation_rounds": multi.conversation_rounds})

        self.run.config.update({"system_prompt": multi.SYSTEM_PROMPT.replace("\n", "\\n")})
        self.run.config.update({"start_prompt": multi.START_PROMPT.replace("\n", "\\n")})
        self.run.config.update({"mid_prompt": multi.MID_PROMPT.replace("\n", "\\n")})
        self.run.config.update({"rules": multi.RULES.replace("\n", "\\n")})
        self.run.config.update({"names": multi.NAMES})
        self.run.config.update({"conversation_rounds": multi.conversation_rounds})

    def log_pipe(self, pipe):
        """
        This function logs all information of the Huggingface pipe.

        @params
            pipe: Huggingface pipeline object
        """
        self.run_data.update({
            "task": pipe.task,
            "model": pipe.model.config.to_dict(),
            "torch_dtype": str(pipe.torch_dtype),
            "device_map": pipe.device.type,
            "do_sample": pipe.generation_config.do_sample,
            "num_beams": pipe.generation_config.num_beams,
            "temperature": pipe.generation_config.temperature,
            "max_new_tokens": pipe.generation_config.max_new_tokens,
            "length_penalty": pipe.generation_config.length_penalty,
            "top_p": pipe.generation_config.top_p,
            "top_k": pipe.generation_config.top_k,
            "min_p": pipe.generation_config.min_p
        })

        self.run.config.update({
            "pipe_task": pipe.task,
            "model": pipe.model.config.to_dict(),
            "torch_dtype": str(pipe.torch_dtype),
            "device_map": pipe.device.type,
            "do_sample": pipe.generation_config.do_sample,
            "num_beams": pipe.generation_config.num_beams,
            "temperature": pipe.generation_config.temperature,
            "max_new_tokens": pipe.generation_config.max_new_tokens,
            "length_penalty": pipe.generation_config.length_penalty,
            "top_p": pipe.generation_config.top_p,
            "top_k": pipe.generation_config.top_k,
            "min_p": pipe.generation_config.min_p
        }, allow_val_change=True)

    def store_run_data(self):
        """
        This function saves the run data in a csv and json file.
        """
        run_dir = "./logs/runs"
        os.makedirs(run_dir, exist_ok=True)

        # assign run_idx
        run_idx = self.get_run_idx()
        self.run.name = str(run_idx)

        # save run data in json file
        with open(f"{run_dir}/{self.run_data['run_idx']}.json", "w", encoding="utf-8") as f:
            json.dump(self.run_data, f, indent=4)

        # save run data in overall csv
        run_dicts = []
        fieldnames = set()
        for run_file in os.listdir(run_dir):
            if run_file.endswith(".json"):
                try:
                    run_dict = json.load(open(f"{run_dir}/{run_file}", 'r', encoding="utf-8"))
                    run_dicts.append(run_dict)
                    fieldnames = fieldnames.union(set(run_dict.keys()))
                except:
                    continue

        run_csv = "./logs/runs.csv"
        with open(run_csv, "w", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")

            writer.writeheader()

            for run_dict in sorted(run_dicts, key= lambda x: x["run_idx"]):
                writer.writerow(run_dict)

        # run_csv = "./logs/runs.csv"
        # if os.path.isfile(run_csv):
        #     prev_runs = pd.read_csv(run_csv, sep = ";")
        #     #curr_run = pd.Series(self.run_data)
        #     all_runs = prev_runs._append(self.run_data, ignore_index = True)
        #     all_runs.to_csv(run_csv, sep = ";", index = False)
        # else:
        #     with open(run_csv, "w", encoding="utf-8") as f:
        #         fieldnames = self.run_data.keys()
        #         writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")

        #         writer.writeheader()
        #         writer.writerow(self.run_data)

    def get_run_idx(self):
        run_dir = "./logs/runs"
        os.makedirs(run_dir, exist_ok=True)
        if "run_idx" not in self.run_data.keys():
            run_files = [int(file.split(".json")[0]) for file in os.listdir(run_dir) if file.endswith(".json")]
            run_idx = max(run_files) + 1
            self.run_data.update({"run_idx": run_idx})
            print(f"Assigning run_idx: {run_idx}\n")
        else: 
            run_idx = self.run_data["run_idx"]
            print(f"Current run_idx: {run_idx}\n")
        
        return run_idx
    
    def print_best_run(self, framework: str = None, split: str = None):
        """
        This function prints the run with the best evaluation result.

        @params
            framework: single- or multi-agent setup
            split: data split
        """
        df = pd.read_csv("./logs/runs.csv", sep = ";")

        filtered_df = df
        
        if framework != None:
            filtered_df = filtered_df.loc[df["framework"] == framework]
        
        if split != None: 
            filtered_df = filtered_df.loc[df["split"] == split]

        filtered_df = filtered_df.mean_acc.dropna()

        if len(filtered_df) != 0:
            df_idx = filtered_df.idxmax()

            print(f"\nBest run for framework {framework} and split {split}:")
            print(df.loc[df_idx])
        else:
            print(f"\nThere are no runs with framework {framework}, split {split} and a duration value!")

    def print_fastest_run(self, framework = None, split = None):
        """
        This function prints the fastest run.

        @params
            framework: single- or multi-agent setup
            split: data split
        """
        df = pd.read_csv("./logs/runs.csv", sep = ";")

        filtered_df = df
        
        if framework != None:
            filtered_df = filtered_df.loc[df["framework"] == framework]
        
        if split != None: 
            filtered_df = filtered_df.loc[df["split"] == split]

        filtered_df = filtered_df.duration.dropna()

        if len(filtered_df) != 0:
            min_df_idx = filtered_df.idxmin()

            print(f"\nFastest run for framework {framework} and split {split}:")
            print(df.loc[min_df_idx])
        else:
            print(f"\nThere are no runs with framework {framework}, split {split} and a duration value!")

if __name__ == "__main__":
    # with open("configs/mini_cpu_multi_exp.json", "r", encoding="utf-8") as f:
    #     config = json.load(f)

    # logger = ExperimentLog(config)
    # logger.store_run_data()
    # logger.print_best_run()

    logger = ExperimentLog()
    # with open("logs/runs/80.json", "r", encoding="utf-8") as f:
    #     logger.run_data = json.load(f)

    # logger.store_run_data()
    logger.print_best_run(framework="multi", split = "dev")