from datasets import load_dataset, Dataset, load_from_disk
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import re
import ast
import dataframe_image as dfi

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

    def plot_instance(self, index: int, setup: str):
        out = self.exp_data_hf[setup].select([index])[0]
        eval = self.eval_data_hf[setup].select([index])[0]

        sent2gold = {sent:idx for idx, sent in enumerate(eval["gold"], start = 1)}
        beliefs = []
        for i, turn in enumerate(out["turns"]):
            conv_list = self.create_conv_list(turn)

            player = out["player"][i]

            for conv_dir in conv_list:
                sent_idx = sent2gold[conv_dir["sentence"]]
                    # Player, sentence, position, time-step
                beliefs.append({
                    "turn": i+1,
                    "agent": player,
                    "gold_sent": sent_idx,
                    "hypo": conv_dir["position"],
                    "correct": sent_idx == conv_dir["position"],
                    "label": f"<{player}, sentence {sent_idx}, hypothesis {conv_dir["position"]}>"
                })

        belief_df = pd.DataFrame(beliefs)

        styled_beliefs = pd.DataFrame(beliefs).style.hide(subset=["correct"], axis=1)
        html = styled_beliefs.to_html()
        with open("test.html", "w", encoding = "utf-8") as file:
            file.write(html)
        #dfi.export(styled_beliefs, "test.png")


    def create_conv_list(self, model_answer):
        conv_str = model_answer.split("</think>")[-1].strip()
        conv_str = re.sub(r"Answer:[ \n]*", "", conv_str).strip()
        # remove newlines to make matching possible
        conv_str = re.sub("\n", "", conv_str)

        # escape single quote strings
        conv_str = re.sub(r'(\b|\b )"(\b| )', r'\1\\"\2', conv_str)
        conv_str = re.sub(r"(\b|\b )'(\b| )", r"\1\\'\2", conv_str)

        # remove json notation
        conv_str = re.sub(r"```json(.*)```", r"\1", conv_str)

        # double quotes
        conv_str = re.sub(r"('|\\\'|\"|\\\"){2,}", r'"', conv_str)
        conv_str = fr"{conv_str}"

        try:
            conv_list = ast.literal_eval(conv_str)
            return conv_list
        except:
            conv_list = []

            sentences = [item[0] for item in re.findall(r"sentence\': \'([\w \.,]+(\\'[\w \.,]+)*)", conv_str)]
            positions = re.findall(r"position\': (\d)", conv_str)
            reason_1_list = [item[0] for item in re.findall(r"reason_1\': \'([\w \.,]+(\\'[\w \.,]+)*)", conv_str)]
            reason_2_list = [item[0] for item in re.findall(r"reason_2\': \'([\w \.,]+(\\'[\w \.,]+)*)", conv_str)]
            reason_3_list = [item[0] for item in re.findall(r"reason_3\': \'([\w \.,]+(\\'[\w \.,]+)*)", conv_str)]
            reason_4_list = [item[0] for item in re.findall(r"reason_4\': \'([\w \.,]+(\\'[\w \.,]+)*)", conv_str)]
            reason_5_list = [item[0] for item in re.findall(r"reason_5\': \'([\w \.,]+(\\'[\w \.,]+)*)", conv_str)]

            for i, sent in enumerate(sentences):
                conv_dict = defaultdict(str)

                conv_dict["sentence"] = sent

                if i < len(positions):
                    conv_dict["positions"] = int(positions[i])
                
                if i < len(reason_1_list):
                    conv_dict["reason_1"] = reason_1_list[i]

                if i < len(reason_2_list):
                    conv_dict["reason_2"] = reason_2_list[i]

                if i < len(reason_3_list):
                    conv_dict["reason_3"] = reason_3_list[i]

                if i < len(reason_4_list):
                    conv_dict["reason_4"] = reason_4_list[i]

                if i < len(reason_5_list):
                    conv_dict["reason_5"] = reason_5_list[i]

                conv_list.append(conv_dict)
            
            return conv_list
        
if __name__ == "__main__":
    # exp_files_no_var = {
    #     "single_hypo": "outputs/single/Qwen3-32B/dev/outputs_146",
    #     "single_full": "outputs/single/Qwen3-32B/dev/outputs_149",
    #     "chain": "outputs/multi/Qwen3-32B/dev/outputs_154",
    #     "star": "outputs/multi/Qwen3-32B/dev/outputs_157"
    # }

    # vis_no_var = Vis(exp_files_no_var, variation = False)
    #vis_no_var.plot_performance()


    exp_files_var = {
        "single_hypo": "outputs/single/Qwen3-32B/dev/outputs_147",
        "single_full": "outputs/single/Qwen3-32B/dev/outputs_152",
        "star": "outputs/multi/Qwen3-32B/dev/outputs_158"
    }

    vis_no_var = Vis(exp_files_var, variation = True)
    #vis_no_var.plot_performance()
    vis_no_var.plot_instance(0, "star")

    print("Done")
