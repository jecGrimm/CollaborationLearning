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

class MultiAgent:
    def __init__(self):
        self.START_PROMPT = '''
        You are {player_name}. You are playing a game with your partners {other_players}. The game has the following rules: {rules}. You start the conversation with the other players. If you have reached a solution, include "STOP" in your answer and return a list of the sentences in the correct order in the following format:\n
        [<sentence1>, <sentence2>, <sentence3>, <sentence4>, <sentence5>]\n
        Answer:\n
        '''

        self.MID_PROMPT='''
        You are {player_name}. You are playing a game with your partners {other_players}. The game has the following rules: {rules}. I\'ll provide the conversation history: {conversation}.  Formulate an adequate response to the ongoing conversation. If you have reached a solution, include "STOP" in your answer and return a list of the sentences in the correct order in the following format:\n\ [<sentence1>, <sentence2>, <sentence3>, <sentence4>, <sentence5>]\n
        Answer:\n
        '''

        self.RULES = '''
        - The game is played by 5 players.
        - Each player receives one sentence that the other players do not know.
        - All sentences together form a short story and are causally and temporally related to each other.
        - The goal of the game is to reassemble the original story by bringing all five sentences into the correct order.

        Your sentence: "{sentence}"
        '''

        self.NAMES = ["Soraya", "Nimrod", "Odett", "Wangui", "Cyrene"]

    def create_message(self, conv_round, turn, inst, conversation = ""):
        message = []

        if conv_round == 0 and turn == 0:
            other_players = self.NAMES.pop(turn)
            rules = self.RULES.format(sentence = inst[f"sentence{turn+1}"])
            message.append(self.START_PROMPT.format(
                player_name = self.NAMES[turn],
                other_players = other_players,
                rules = rules
            ))
        else:
            other_players = self.NAMES.pop(turn)
            rules = self.RULES.format(sentence = inst[f"sentence{turn+1}"])
            message.append(self.MID_PROMPT.format(
                player_name = self.NAMES[turn],
                other_players = other_players,
                rules = rules,
                conversation = conversation
            ))

        return message

    def play_games(self, batch, pipe):
        batch_convs = []
        for inst in batch:
            conv_round = 0
            conversation = ""
            while conv_round < 4:
                for turn in range(5):
                    model_out = pipe(
                            self.create_message(conv_round, turn, inst, conversation),
                            max_new_tokens=256,
                        )
                    
                    model_answer = model_out[0][0]["generated_text"].split("Answer:\n")[-1].strip()
                
                conversation += f"\n{self.NAMES[turn]}: {model_answer}\n"

                if "STOP" in model_answer:
                    conv_round = 4
                else:
                    conv_round += 1 
            
            batch_convs.append(conversation)

        return {"outputs": batch_convs}
                

    def prompt_models(self, split = "mini", device = "cpu"):
        model_id = "meta-llama/Llama-3.2-1B-Instruct"
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
    
        roc_data = ROCStories(split = split)
        prompt_data = roc_data.orig_dataset
        outputs = prompt_data.map(lambda batch: self.play_games(batch, pipe), batched=True)

        model_name = model_id.split("/")[-1]
        os.makedirs(f"./outputs/multi/{model_name}/{split}", exist_ok = True)
        outfile = f"./outputs/multi/{model_name}/{split}/outputs_multi_{model_name}_{split}_{str(datetime.now()).replace(" ", "_")}.jsonl"
        outputs.to_json(outfile, indent=4)

        return outfile