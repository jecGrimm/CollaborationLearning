from datasets import load_dataset, Dataset, load_from_disk
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import re
import ast
from pySankey.sankey import sankey
import numpy as np
from evaluate import Evaluator
#import plotly.graph_objects as go # TODO: Full sankey with go https://python-graph-gallery.com/sankey-diagram-with-python-and-plotly/

class Vis():
    def __init__(self, experiment_files: dict, variation: bool, conv_format: str, last: bool):
        self.experiment_files = experiment_files

        self.last = last

        self.conv_format = conv_format
        self.variation = variation
        if self.variation:
            self.vispath = f"visualizations/variation/{conv_format}"
        else:
            self.vispath = f"visualizations/no_variation/{conv_format}"
        os.makedirs(self.vispath, exist_ok = True)

        self.exp_data_hf = {k: load_from_disk(v) for k, v in experiment_files.items()}

        if last:
            eval_files = defaultdict(str)
            for k, v in experiment_files.items():
                if "single" in k:
                    eval_files[k] = f"{v.replace("outputs", "eval")}.jsonl"
                else:
                    eval_files[k] = f"{v.replace("outputs", "eval")}_last.jsonl"
            #eval_files = {k: f"{v.replace("outputs", "eval")}_last.jsonl" for k, v in experiment_files.items()}
        else:
            eval_files = {k: f"{v.replace("outputs", "eval")}.jsonl" for k, v in experiment_files.items()}
        self.eval_data_hf = {k: self.load_eval_data(v, k) for k, v in eval_files.items()}

        mean_preds = defaultdict(list)
        self.metrics = ["accuracy", "prec_1", "prec_2", "prec_3", "prec_4", "prec_5", "spearman", "kendalltau"]

        for setup, eval_data in self.eval_data_hf.items():
            for metric in self.metrics:
                mean_preds[setup].append(float(eval_data.with_format("pandas")[metric].mean()))

        self.mean_preds_df = pd.DataFrame(mean_preds.values(), index=mean_preds.keys(), columns=self.metrics) 

    def load_eval_data(self, eval_file, exp_name):
        try:
            return load_dataset("json", data_files=eval_file, split="train")
        except:
            fw = ""
            if "single" in eval_file:
                fw = "single"
            else:
                fw = "multi"

            evaluator = Evaluator(self.experiment_files[exp_name], fw, last = self.last)
            
            if self.last and fw == "multi":
                evaluator.evaluate_last_beliefs()
            else:
                evaluator.evaluate_sent_order()
            return load_dataset("json", data_files=eval_file, split="train")

    def plot_performance(self):
        for metric in self.metrics:
            if self.variation:
                ax = self.mean_preds_df[metric].plot(kind = "bar", title= f"Reordering: with variation", rot=0, ylim=[0.0, 1.0], ylabel=metric) 

                ax.bar_label(ax.containers[0])

                if self.last:
                    plt.savefig(f"{self.vispath}/{metric}_var_{self.conv_format}_last.png")
                else:
                    plt.savefig(f"{self.vispath}/{metric}_var_{self.conv_format}.png")
            else:
                ax = self.mean_preds_df[metric].plot(kind = "bar", title= f"Reordering: without variation", rot=0, ylim=[0.0, 1.0], ylabel=metric) 
                ax.bar_label(ax.containers[0])
                
                if self.last:
                    plt.savefig(f"{self.vispath}/{metric}_no_var_{self.conv_format}_last.png")
                else:
                    plt.savefig(f"{self.vispath}/{metric}_no_var_{self.conv_format}.png")

    def create_belief_instance(self, index: int, setup: str):
        out = self.exp_data_hf[setup].select([index])[0]
        eval = self.eval_data_hf[setup].select([index])[0]

        sent2gold = {sent:idx for idx, sent in enumerate(eval["gold"], start = 1)}
        beliefs = []
        for i, turn in enumerate(out["turns"]):
            conv_list = self.create_conv_list(turn)

            player = out["player"][i]

            for conv_dir in conv_list:
                if conv_dir["sentence"] in sent2gold.keys() and "position" in conv_dir.keys():
                    sent_idx = sent2gold[conv_dir["sentence"]]
                    
                    # Player, sentence, position, time-step
                    if sent_idx == conv_dir["position"]:
                        correct = "correct"
                    else:
                        correct = "incorrect"

                    beliefs.append({
                        "turn": i+1,
                        "agent": player,
                        "gold_sent": sent_idx,
                        "hypo": conv_dir["position"],
                        "correct": correct,
                        "label": f"({player}, sentence {sent_idx}, hypothesis {conv_dir["position"]})"
                    })

        return beliefs
        

    def plot_instance(self, index: int, setup: str):

        #vis_beliefs_df = pd.DataFrame(vis_beliefs)
        beliefs = self.create_belief_instance(index, setup)
        belief_df = pd.DataFrame(beliefs)
        belief_pivot = belief_df.pivot(columns = ["correct"], values=["turn", "label"])

        styler = belief_pivot.style
        styler.hide(axis = 0)\
            .highlight_null(props="color: transparent;")\
            .set_properties(props="color: transparent;")

        # styled_beliefs = styled_beliefs.style.highlight_null(props="color: transparent;")
        # styled_beliefs = styled_beliefs.hide_index()
        #styled_beliefs = styled_beliefs.set_index("turn")

        html = styler.to_html()
        with open(f"{self.vispath}/instance.html", "w", encoding = "utf-8") as file:
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

            sentences = [item[0] for item in re.findall(r"""sentence[\'"]: [\'"]([\w \.,]+(\\'[\w \.,]+)*)""", conv_str)]
            positions = re.findall(r"""position[\'"]: (\d)""", conv_str)
            reason_1_list = [item[0] for item in re.findall(r"""reason_1[\'"]: [\'"]([\w \.,]+(\\'[\w \.,]+)*)""", conv_str)]
            reason_2_list = [item[0] for item in re.findall(r"""reason_2[\'"]: [\'"]([\w \.,]+(\\'[\w \.,]+)*)""", conv_str)]
            reason_3_list = [item[0] for item in re.findall(r"""reason_3[\'"]: [\'"]([\w \.,]+(\\'[\w \.,]+)*)""", conv_str)]
            reason_4_list = [item[0] for item in re.findall(r"""reason_4[\'"]: [\'"]([\w \.,]+(\\'[\w \.,]+)*)""", conv_str)]
            reason_5_list = [item[0] for item in re.findall(r"""reason_5[\'"]: [\'"]([\w \.,]+(\\'[\w \.,]+)*)""", conv_str)]

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
        
    def plot_sankey(self, setup: str):
        # last hypo == correct/incorrect -> final hypo == correct/incorrect 
        self.sankey_inst = []
        self.exp_data_hf[setup].map(lambda x, idx: self.create_sankey_hf(x, idx, setup), with_indices=True)

        #df = pd.read_csv(url, sep=" ", names=["true", "predicted"])
        df = pd.DataFrame(self.sankey_inst, columns = ["last_hypo", "final_hypo"])

        colors = {
            "correct": "#f78c1b",
            "incorrect": "#1b7ef7",
        }

        sankey(df["last_hypo"], df["final_hypo"], aspect=20, colorDict=colors, fontsize=12)

        fig = plt.gcf()

        if self.variation:
            fig.savefig(f"{self.vispath}/sankey_{setup}.png", bbox_inches="tight", dpi=150)
        else:
            fig.savefig(f"{self.vispath}/sankey_{setup}.png", bbox_inches="tight", dpi=150)

    def create_sankey_hf(self, x, idx, setup):
        beliefs = self.create_belief_instance(idx, setup)

        seen_sents = []
        for belief in beliefs[::-1]:
            if belief["agent"] != "Player 6" and belief["gold_sent"] not in seen_sents:
                for final_sent_belief in beliefs[::-1]:
                    if final_sent_belief["agent"] == "Player 6" and final_sent_belief["gold_sent"] == belief["gold_sent"]:
                        self.sankey_inst.append([belief["correct"], final_sent_belief["correct"]])

    def create_full_sankey_hf(self, x, idx, setup):
        beliefs = self.create_belief_instance(idx, setup)

        sent_beliefs = [["not mentioned" for _ in range(self.num_turns)] for _ in range(5)]

        turn_idx = 0
        for belief in beliefs:
            if setup == "chain":
                turn_idx = belief["turn"]-1
            elif setup == "star":
                if belief["agent"] != "Player 6":
                    turn_idx = 0
                else:
                    turn_idx = 1
            elif setup == "graph":
                if belief["turn"] < 6:
                    turn_idx = 0
                elif belief["turn"] > 6 and belief["agent"] != "Player 6":
                    turn_idx = 1
                else:
                    turn_idx = 2

            sent_beliefs[belief["gold_sent"]-1][turn_idx] = belief["correct"]

            # TODO:graph
        self.sankey_inst += sent_beliefs

    def plot_sankey_full(self, setup: str):
        # last hypo == correct/incorrect -> final hypo == correct/incorrect 

        #df = pd.read_csv(url, sep=" ", names=["true", "predicted"])
        self.num_turns = 0
        if setup == "chain":
            turn_lengths = [len(self.exp_data_hf[setup]["player"][i]) for i in range(len(self.exp_data_hf[setup]["player"]))]
            self.num_turns = max(turn_lengths)
        elif setup == "star":
            self.num_turns = 2
        elif setup == "graph":
            self.num_turns = 3

        columns = [f"Turn {i}" for i in range(1, self.num_turns+1)]
        self.sankey_inst = []

        self.exp_data_hf[setup].map(lambda x, idx: self.create_full_sankey_hf(x, idx, setup), with_indices=True)
        
        df = pd.DataFrame(self.sankey_inst, columns = columns) # TODO: define column names

        colors = {
            "correct": "#f78c1b",
            "incorrect": "#1b7ef7",
            "not mentioned": "#f5ee27"
        }

        sankey(df["last_hypo"], df["final_hypo"], aspect=20, colorDict=colors, fontsize=12)

        fig = plt.gcf()

        if self.variation:
            fig.savefig(f"{self.vispath}/sankey_{setup}_var__{self.conv_format}.png", bbox_inches="tight", dpi=150)
        else:
            fig.savefig(f"{self.vispath}/sankey_{setup}_no_var__{self.conv_format}.png", bbox_inches="tight", dpi=150)

        
if __name__ == "__main__":
    # full-full-no_var
    # exp_files_no_var = {
    #     "single_hypo": "outputs/single/Qwen3-32B/dev/outputs_146",
    #     "single_full": "outputs/single/Qwen3-32B/dev/outputs_149",
    #     "chain": "outputs/multi/Qwen3-32B/dev/outputs_154",
    #     "graph": "outputs/multi/Qwen3-32B/dev/outputs_167",
    #     "star": "outputs/multi/Qwen3-32B/dev/outputs_157"
    # }

    # vis_no_var = Vis(exp_files_no_var, variation = False, conv_format = "full", last = False)
    # vis_no_var.plot_performance()
    # #vis_no_var.plot_sankey_full("graph")
    # vis_no_var.plot_sankey("graph")

    # full-full_var
    # exp_files_var = {
    #     "single_hypo": "outputs/single/Qwen3-32B/dev/outputs_147",
    #     "single_full": "outputs/single/Qwen3-32B/dev/outputs_152",
    #     "chain_750": "outputs/multi/Qwen3-32B/dev/ckps/outputs_155_0_1_173_0",
    #     "graph_250": "outputs/multi/Qwen3-32B/dev/ckps/outputs_176_0",
    #     "star": "outputs/multi/Qwen3-32B/dev/outputs_158",
    # }

    # vis_var = Vis(exp_files_var, variation = True, conv_format = "full", last = False)
    # vis_var.plot_performance()
    # #vis_var.plot_instance(0, "star")
    # vis_var.plot_sankey("graph_250")

    # full-hypo: no_var
    # exp_files_full_hypo_no_var = {
    #     "single_hypo": "outputs/single/Qwen3-32B/dev/outputs_146",
    #     "single_full": "outputs/single/Qwen3-32B/dev/outputs_149",
    #     "chain": "outputs/multi/Qwen3-32B/dev/outputs_163"
    # }

    # vis_full_hypo_no_var = Vis(exp_files_full_hypo_no_var, variation = False, conv_format = "hypo", last = True)
    # vis_full_hypo_no_var.plot_performance()
    # vis_full_hypo_no_var.plot_sankey("star")

    # var: full-hypo
    exp_files_full_hypo_var = {
        "single_hypo": "outputs/single/Qwen3-32B/dev/outputs_147",
        "single_full": "outputs/single/Qwen3-32B/dev/outputs_152",
        "chain_500": "outputs/multi/Qwen3-32B/dev/ckps/outputs_169_0_177_0",
        "star": "outputs/multi/Qwen3-32B/dev/outputs_179"
    }

    vis_full_hypo_var = Vis(exp_files_full_hypo_var, variation = True, conv_format = "hypo", last = False)
    vis_full_hypo_var.plot_performance()
    #vis_full_hypo_var.plot_sankey("star")

    print("Done")


