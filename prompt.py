import torch
from transformers import pipeline
from data import ROCStories
import os
from datetime import datetime

class SingleAgent:
    def __init__(self):
        self.PROMPT = '''\
            Please bring the following sentences in the correct order based on their causal and temporal relations:\n\
                {shuffled_sents}\n\
            Return a list of the sentences in the correct order in the following format:\n\
                [<sentence1>, <sentence2>, <sentence3>, <sentence4>, <sentence5>]\n\
            Answer:\n\
            '''

    def create_message(self, batch):
        messages = []
        for i in range(len(batch["storyid"])):
            #messages.append({"role": "user", "content": prompt.format(shuffled_sents = batch["shuffled_sents"][i])})
            messages.append(self.PROMPT.format(shuffled_sents = batch["shuffled_sents"][i]))

        return {"prompt": messages}
    
    def prompt_model(self, split = "mini", device = "cpu"):
        model_id = "meta-llama/Llama-3.2-1B-Instruct"
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
    
        roc_data = ROCStories(split = split)
        prompt_data = roc_data.shuffled_dataset.map(self.create_message, batched=True)

        outputs = pipe(
            list(prompt_data["prompt"]),
            max_new_tokens=256,
        )
        print(outputs[0][0]["generated_text"])

        output_list = [turn["generated_text"] for conv in outputs for turn in conv]
        num_missing = len(prompt_data) - len(output_list)
        output_list = output_list + [""] * num_missing

        output_data = prompt_data.add_column("output", output_list)

        model_name = model_id.split("/")[-1]
        os.makedirs(f"./outputs/single/{model_name}/{split}", exist_ok = True)
        outfile = f"./outputs/single/{model_name}/{split}/outputs_single_{model_name}_{split}_{str(datetime.now()).replace(" ", "_")}.jsonl"
        output_data.to_json(outfile, indent=4)

        return outfile
