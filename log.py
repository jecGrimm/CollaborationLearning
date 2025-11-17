from collections import defaultdict
import os
import json
import csv
import pandas as pd
import re

class ExperimentLog:
    def __init__(self, config = None):
        # self.run = wandb.init(
        #     # Set the wandb entity where your project will be logged (generally your team name).
        #     entity="j-grimm-ludwig-maximilian-university-of-munich",
        #     # Set the wandb project where this run will be logged.
        #     project="CollaborationLearning",
        #     # Track hyperparameters and run metadata.
        #     config=config
        # )

        # self.output_artifact = wandb.Artifact(name="output", type="dataset")
        
        self.run_data = defaultdict()

        if config:
            self.run_data.update(config)

    def log_single(self, single):
        self.run_data.update({"user_prompt": single.PROMPT.replace("\n", "\\n")})

    def log_multi(self, multi):
        self.run_data.update({"system_prompt": multi.SYSTEM_PROMPT.replace("\n", "\\n")})
        self.run_data.update({"start_prompt": multi.START_PROMPT.replace("\n", "\\n")})
        self.run_data.update({"mid_prompt": multi.MID_PROMPT.replace("\n", "\\n")})
        self.run_data.update({"rules": multi.RULES.replace("\n", "\\n")})
        self.run_data.update({"names": multi.NAMES})

    def log_pipe(self, pipe):
        self.run_data.update({
            "task": pipe.task,
            "model": pipe.model.config.to_dict(),
            "torch_dtype": str(pipe.torch_dtype),
            "device_map": pipe.device.type
        })

    def store_run_data(self):
        run_dir = "./logs/runs"
        os.makedirs(run_dir, exist_ok=True)

        if "run_idx" not in self.run_data.keys():
            run_files = [int(file.split(".json")[0]) for file in os.listdir(run_dir) if file.endswith(".json")]
            run_idx = max(run_files) + 1
            self.run_data.update({"run_idx": run_idx})

        with open(f"{run_dir}/{self.run_data["run_idx"]}.json", "w", encoding="utf-8") as f:
            json.dump(self.run_data, f, indent=4)

        run_csv = "./logs/runs.csv"
        if os.path.isfile(run_csv):
            prev_runs = pd.read_csv(run_csv, sep = ";")
            #curr_run = pd.Series(self.run_data)
            all_runs = prev_runs._append(self.run_data, ignore_index = True)
            all_runs.to_csv(run_csv, sep = ";", index = False)
        else:
            with open(run_csv, "w", encoding="utf-8") as f:
                fieldnames = self.run_data.keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")

                writer.writeheader()
                writer.writerow(self.run_data)
    
    def print_best_run(self, framework = None):
        df = pd.read_csv("./logs/runs.csv", sep = ";")
        if framework == None:
            max_df_idx = df.mean_acc.idxmax()
        else:
            max_df_idx = df.loc[df["framework"] == framework].mean_acc.idxmax()
        print(df.loc[max_df_idx])

    def print_fastest_run(self, framework = None, split = None):
        df = pd.read_csv("./logs/runs.csv", sep = ";")

        filtered_df = df
        
        if framework != None:
            filtered_df = filtered_df.loc[df["framework"] == framework]
        
        if split != None: 
            filtered_df = filtered_df.loc[df["split"] == split]

        filtered_df = filtered_df.duration.dropna()

        if len(filtered_df) != 0:
            min_df_idx = filtered_df.duration.idxmin()

            print(df.loc[min_df_idx])
        else:
            print(f"There are no runs with framework {framework}, split {split} and a duration value!")

if __name__ == "__main__":
    # with open("configs/mini_cpu_multi_exp.json", "r", encoding="utf-8") as f:
    #     config = json.load(f)

    # logger = ExperimentLog(config)
    # logger.store_run_data()
    # logger.print_best_run()

    logger = ExperimentLog()
    logger.print_fastest_run(framework="single", split = "dev")