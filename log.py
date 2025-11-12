from collections import defaultdict
import os
import json
import csv
import pandas as pd

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
        self.run_data.update({"user_prompt", single.PROMPT})

    def log_multi(self, multi):
        self.run_data.update({"system_prompt", multi.SYSTEM_PROMPT})
        self.run_data.update({"start_prompt", multi.START_PROMPT})
        self.run_data.update({"mid_prompt", multi.MID_PROMPT})
        self.run_data.update({"rules", multi.RULES})
        self.run_data.update({"names", multi.NAMES})

    def log_pipe(self, pipe):
        self.run_data.update({
            "task": pipe.task,
            "model": pipe.model,
            "torch_dtype": pipe.torch_dtype,
            "device_map": pipe.device_map
        })

    def store_run_data(self):
        run_dir = "./logs/runs"
        os.makedirs(run_dir, exist_ok=True)

        run_idx = len(os.listdir(run_dir)) + 1
        self.run_data.update({"run_idx": run_idx})

        with open(f"{run_dir}/{run_idx}.json", "w", encoding="utf-8") as f:
            json.dump(self.run_data, f, indent=4)

        run_csv = "./logs/runs.csv"
        if os.path.isfile(run_csv):
            prev_runs = pd.read_csv(run_csv, sep = ";")
            curr_run = pd.Series(self.run_data)
            prev_runs._append(curr_run, ignore_index = True)
            prev_runs.to_csv(run_csv, sep = ";")
        else:
            with open(run_csv, "w", encoding="utf-8") as f:
                fieldnames = self.run_data.keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                writer.writeheader()
                writer.writerow(self.run_data)
    
    def print_best_run(self):
        df = pd.read_csv("./logs/runs.csv", sep = ";")
        max_df_idx = df.mean_acc.idxmax()
        print(df.loc[max_df_idx])

    def print_fastest_run(self):
        df = pd.read_csv("./logs/runs.csv", sep = ";")
        min_df_idx = df.duration.idxmin()
        print(df.loc[min_df_idx])

if __name__ == "__main__":
    with open("configs/mini_cpu_multi_exp.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    logger = ExperimentLog(config)
    logger.store_run_data()
    logger.print_best_run()