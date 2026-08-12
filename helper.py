import csv 
from datasets import load_dataset, load_from_disk, concatenate_datasets
import random
from evaluate import evaluate

def check_left_insts(outdir: str):
    """
    This function checks how many instances are still part of the ongoing conversation in the multi-agent framework (= how many outputs do not contain "STOP" in the final answer).

    @params
        outdir: path to the output dataset
    """
    exp_data = load_from_disk(outdir)

    filtered_data = exp_data.filter(lambda batch: [batch["stop"][i] == False for i in range(len(batch["storyid"]))], batched = True)

    print(len(filtered_data))

def inspect_instances(outdir: str, rand_idx: int = None, outfile: str = None):
    """
    This function prints the output and evaluation results for a random or specified instance.

    @params
        outdir: path to output dataset
        rand_idx: idx of the instance, random if None
    """
    # load output data
    exp_data = load_from_disk(outdir)
    
    # load evaluation data
    eval_file = outdir.replace("outputs", "eval") + ".jsonl"
    eval_data = load_dataset("json", data_files=eval_file, split="train")
    
    # assign random idx
    if rand_idx == None:
        rand_idx = random.randint(0, len(eval_data)-1)

    # select instances
    out = exp_data.select([rand_idx])[0]
    eval = eval_data.select([rand_idx])[0]

    # print information 
    outstr = f"Output File: {outdir}\nIDX: {rand_idx}\n"
    for k, v in out.items():
        if k == "turns":
            outstr += f"\n{k}:\n\n"
            for turn in v:
                outstr += f"\n{turn}\n"
        else:
            outstr += f"\n{k}:\n{v}\n"
    for k,v in eval.items():
        outstr += f"\n{k}:\n{v}\n"
    
    if outfile:
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(outstr)
    else:
        print(outstr)

def merge_outputs(outdirs: list, outpath: str, framework: str, task: str, structured: bool, data_len: int = 1000):
    '''
    This function merges multiple output datasets into one and evaluates the merged dataset.

    @params
        outdirs: list of paths to the output datasets to be merged
        outpath: path to save the merged dataset
        framework: "single" or "multi"
        task: task name (e.g., "ROCStories", "CMUBookSum")
        structured: boolean indicating whether the data is structured
        data_len: maximum number of instances to keep in the merged dataset (default is 1000)
    '''
    exp_data = [load_from_disk(outdir) for outdir in outdirs]

    if len(exp_data) > data_len:
        full_data = concatenate_datasets(exp_data).select(range(1500))
    else:
        full_data = concatenate_datasets(exp_data)

    print("New data length: ", len(full_data))

    full_data.save_to_disk(outpath)

    evaluate(outpath, framework, task, structured=structured, all=True)

def fix_outputs(outdir):
    '''
    This function fixes the outputs in the dataset by splitting the "thinking_output" field and keeping only the part after the last "</think>" tag.

    @params
        outdir: path to the output dataset to be fixed
    '''
    exp_data = load_from_disk(outdir)

    exp_data = exp_data.map(split_think)

    exp_data.save_to_disk(f"{outdir}_fixed")

def split_think(inst):
    '''
    This function splits the "thinking_output" field of an instance and keeps only the part after the last "</think>" tag.

    @params
        inst: a single instance from the dataset
    @returns
        dict: a dictionary with the updated "output" field
    '''
    return {"output": inst["thinking_output"].split("</think>")[-1].strip()}

if __name__ == "__main__":
    conv_output = "outputs/multi/Qwen3-32B/rocstories/test/outputs_451"
    conv_output1 = "outputs/multi/Qwen3-32B/rocstories/test/outputs_452"
    outfile = "outputs/multi/Qwen3-32B/rocstories/test/outputs_451_452"

    #inspect_instances(conv_output, rand_idx=2, outfile=outfile)

    merge_outputs([conv_output, conv_output1], outfile, framework="multi", task="rocstories", structured=True)