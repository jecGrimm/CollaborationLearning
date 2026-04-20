def get_instances(outdir: str, rand_idx: int = None):
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