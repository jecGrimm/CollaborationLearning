from datasets import load_dataset, Dataset, load_from_disk
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import os

class Vis():
    def __init__(self, experiment_files: dict, variation: bool):
        self.variation = variation
        if self.variation:
            os.makedirs("visualizations/variation", exist_ok = True)
        else:
            os.makedirs("visualizations/no_variation", exist_ok = True)

        self.exp_data_hf = {k: load_from_disk(v) for k, v in experiment_files.items()}

        eval_files = {k: f"{v.replace("outputs", "eval")}.jsonl" for k, v in experiment_files.items()}
        self.eval_data_hf = {k: load_dataset("json", data_files=v, split="train") for k, v in eval_files.items()}

        mean_preds = defaultdict(list)
        self.metrics = ["accuracy", "prec_1", "prec_2", "prec_3", "prec_4", "prec_5", "spearman", "kendalltau"]

        for setup, eval_data in self.eval_data_hf.items():
            for metric in self.metrics:
                mean_preds[setup].append(float(eval_data.with_format("pandas")[metric].mean()))

        self.mean_preds_df = pd.DataFrame(mean_preds.values(), index=mean_preds.keys(), columns=self.metrics) 


    def plot_performance(self):
        for metric in self.metrics:
            if self.variation:
                self.mean_preds_df[metric].plot(kind = "bar", title= f"Reordering: with variation", rot=0, ylim=[0.0, 1.0], ylabel=metric) 

                plt.savefig(f"visualizations/variation/{metric}_var.png")
            else:
                self.mean_preds_df[metric].plot(kind = "bar", title= f"Reordering: without variation", rot=0, ylim=[0.0, 1.0], ylabel=metric) 

                plt.savefig(f"visualizations/no_variation/{metric}_no_var.png")

if __name__ == "__main__":
    exp_files_no_var = {
        "single_hypo": "outputs/single/Qwen3-32B/dev/outputs_146",
        "single_full": "outputs/single/Qwen3-32B/dev/outputs_149",
        "chain": "outputs/multi/Qwen3-32B/dev/outputs_154",
        "star": "outputs/multi/Qwen3-32B/dev/outputs_157"
    }

    vis_no_var = Vis(exp_files_no_var, variation = False)
    vis_no_var.plot_performance()


    exp_files_var = {
        "single_hypo": "outputs/single/Qwen3-32B/dev/outputs_147",
        "single_full": "outputs/single/Qwen3-32B/dev/outputs_152",
        "star": "outputs/multi/Qwen3-32B/dev/outputs_158"
    }

    vis_no_var = Vis(exp_files_var, variation = True)
    vis_no_var.plot_performance()

    print("Done")
