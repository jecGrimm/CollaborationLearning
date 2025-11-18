from prompt import SingleAgent, MultiAgent
import os 
from cli import CLI
from evaluate import evaluate_sent_order
from log import ExperimentLog
import pandas as pd
import torch

def run_experiment(config = None, logger = None):
    if logger == None:
        logger = ExperimentLog(config)

    outfile = config["experiment_file"]

    if not (os.path.isfile(outfile) or os.path.isdir(outfile)):
        # prompt

        if config["framework"] == "single":
            single = SingleAgent(model_id=config["model_id"], split=config["split"])

            if "user_prompt" in config.keys():
                single.PROMPT = config["user_prompt"]
            
            logger.log_single(single)

            single.prompt_model(logger, device = config["device"], batch_size = config["batch_size"]) 

            outfile = single.define_outfile(hf = True)
        else:
            multi = MultiAgent(model_id=config["model_id"], split=config["split"])

            logger.log_multi(multi)

            multi.prompt_models(logger, device = config["device"], batch_size = config["batch_size"])

            outfile = multi.define_outfile(hf = True)

    # evaluate
    evaluate_sent_order(outfile, config["framework"], logger)

    logger.store_run_data()

def test_batch_size(split = "mini", device = "cpu", batch_sizes = [1, 8, 64, 256, 1000], frameworks = ["single", "multi"]):
    for fw in frameworks:
        print("\n")
        print("-" * 30)
        print(f"Processing framework {fw}")
        for batch_size in batch_sizes:
            print("\n")
            print(f"Processing batch size {batch_size}")
            config = {
                "split": split,
                "device": device,
                "framework": fw,
                "experiment_file": "",
                "batch_size": batch_size,
                "model_id": "meta-llama/Llama-3.2-1B-Instruct"
            }
            logger = ExperimentLog(config)

            run_experiment(config, logger)
            print(f"Duration: {logger.run_data["duration"]}")

        print("\n")
        logger.print_fastest_run(framework=fw, split = split)

def test_single_llm_rewritten_prompts(split = "mini", device = "cpu"):
    prompt_candidates = {
        "gemini-2.5-pro": """\
            Arrange the following sentences into a coherent paragraph, ordered by time and logic.\n\
            \n\
            Sentences to order:\n\
            {shuffled_sents}\n\
            \n\
            Return your answer as a single Python list of strings.\n\
            Answer:\n\
            """,
        "llama-3.3-70b-instruct": '''\
            Reorder the following sentences to reflect their logical causal and temporal relationships. Please arrange the sentences in a coherent and chronological order, ensuring that the events described are in a logical sequence. The reordered sentences should be returned as a list in the format:\n\
            [<sentence1>, <sentence2>, <sentence3>, <sentence4>, <sentence5>]\n\
            where <sentence1> is the first event in the sequence, <sentence2> is the second event, and so on.\n\
            The sentences to be reordered are:\n\
            {shuffled_sents}\n\
            Please provide the reordered list of sentences.\n\
            Answer:\n\
            '''
    }

    for model, prompt in prompt_candidates.items():
        print("\n")
        print("-" * 30)
        print("Processing prompt written by ", model)
        config = {
                "split": split,
                "device": device,
                "framework": "single",
                "experiment_file": "",
                "batch_size": 256,
                "model_id": "meta-llama/Llama-3.2-1B-Instruct",
                "user_prompt": prompt
            }
        logger = ExperimentLog(config)
        run_experiment(config, logger)

        print("Accuracy:", logger.run_data["mean_acc"])

    print("\n")
    logger.print_best_run(framework = "single")
        
if __name__ == "__main__":
    cli = CLI()
    # run_experiment(cli.config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    test_batch_size(split = cli.config["split"], device = device, batch_sizes=cli.config["batch_sizes"], frameworks=cli.config["frameworks"])

    #test_batch_size(split = "dev", device = device, batch_sizes=[256], frameworks=["multi"])

    # config = {
    #     "split": "dev",
    #     "device": device,
    #     "framework": "single",
    #     "experiment_file": "",
    #     "model_id": "meta-llama/Llama-3.2-1B-Instruct",
    #     "batch_size": 8
    # }
    # run_experiment(config)

    #test_single_llm_rewritten_prompts(split = "mini", device="device")