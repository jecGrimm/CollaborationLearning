from prompt import SingleAgent, MultiAgent
import os 
from cli import CLI
from evaluate import evaluate_sent_order
from log import ExperimentLog
import pandas as pd

def run_experiment(config = None, logger = None):
    if logger == None:
        logger = ExperimentLog(config)

    outfile = config["experiment_file"]

    if not os.path.isfile(outfile):
        # prompt

        if config["framework"] == "single":
            single = SingleAgent(model_id=config["model_id"], split=config["split"])
            
            logger.log_single(single)

            single.prompt_model(logger, device = config["device"], batch_size = config["batch_size"]) 

            outfile = single.define_outfile()
        else:
            multi = MultiAgent(model_id=config["model_id"], split=config["split"])

            logger.log_multi(multi)

            multi.prompt_models(logger, device = config["device"], batch_size = config["batch_size"])

            outfile = multi.define_outfile()

    # evaluate
    evaluate_sent_order(outfile, config["framework"], logger)

    logger.store_run_data()

def test_batch_size(split = "mini", device = "cpu"):
    for fw in ["single", "multi"]:
        print(f"Processing framework {fw}")
        for batch_size in [1, 8, 64, 256, 1000]:
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

        logger.print_fastest_run()
        
if __name__ == "__main__":
    # cli = CLI()
    # run_experiment(cli.config)

    test_batch_size(split = "mini", device = "cuda:4")