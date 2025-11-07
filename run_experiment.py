from prompt import SingleAgent
import os 
from cli import CLI
from evaluate import evaluate_sent_order

cli = CLI()
outfile = cli.experiment_file

if not os.path.isfile(cli.experiment_file):
    # prompt
    single = SingleAgent()
    outfile = single.prompt_model(split = cli.split, device = cli.device) # TODO: Check if new file is created if it's already available

# evaluate
evaluate_sent_order(outfile)
