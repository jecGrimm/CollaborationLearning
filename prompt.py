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
        self.SYSTEM_PROMPT = '''
        I am {player_name}, playing a game with my partners {other_players}. The game has the following rules: {rules}. At each time step, I will provide you with the relevant information of the game. I will also provide you with the conversation history up to this point. Please help me to formulate an adequate response to the conversation. If all five sentences have been collected during the conversation, assemble them in the correct order, include "STOP" in your answer and return a list of the sentences in the correct order in the following format:  
        [<sentence1>, <sentence2>, <sentence3>, <sentence4>, <sentence5>]
        '''

        self.START_PROMPT = '''It is currently my ({player_name}) turn and my sentence is "{sentence}". I start the conversation with the other players ({other_players}). Please formulate a beginning statement for the conversation for me. Do not say anything else. Got it?
        Answer:
        '''
        self.MID_PROMPT='''It is currently my ({player_name}) turn and my sentence is "{sentence}". I continue the conversation with the other players ({other_players}).
        The conversation history: 
        {conversation}
        Please formulate a response to the conversation for me. Do not say anything else. Got it?
        Answer:
        '''

        self.RULES = '''
        - The game is played by 5 players.
        - Each player receives one sentence that the other players do not know.
        - All sentences together form a short story and are causally and temporally related to each other.
        - The goal of the game is to reassemble the original story by bringing all five sentences into the correct order.
        - The players conduct a conversation where they exchange their sentences and discuss how they could be related to each other and how they could be ordered correctly.
        
        My sentence: "{sentence}"
        '''

        self.NAMES = ["Soraya", "Nimrod", "Odett", "Wangui", "Cyrene"]

    def create_message(self, conv_round, turn, conversation, roles, player_name):
        other_players = [name for name in self.NAMES if name != player_name]
        player_sent = roles[player_name]
        rules = self.RULES.format(sentence = player_sent)

        message = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT.format(
                    player_name = player_name,
                    other_players = other_players,
                    rules = rules
                    )
            }
        ]

        if conv_round == 0 and turn == 0:
            message.append(
                {
                    "role": "user",
                    "content": self.START_PROMPT.format(
                    player_name = player_name,
                    other_players = other_players,
                    sentence = player_sent
                    )
                }
            )
        else:
            message.append(
                {
                    "role": "user",
                    "content": self.MID_PROMPT.format(
                        player_name = player_name,
                        other_players = other_players,
                        conversation = conversation,
                        sentence = player_sent
                    )
                }
            )

        return message

    def play_games(self, batch, pipe):
        batch_convs = []
        for i in range(len(batch["storyid"])):
            conv_round = 0
            turn = 0
            conversation = ""
            turns = []

            roles = dict(zip(self.NAMES, batch["shuffled_sents"][i]))
            while conv_round < 4:
                while turn < 5:
                    player_name = self.NAMES[turn]

                    model_out = pipe(
                            self.create_message(conv_round, turn, conversation, roles, player_name),
                            max_new_tokens=256,
                        )
                    
                    model_answer = model_out[0]["generated_text"][2]["content"]
                
                    conversation += f"\n{player_name}: {model_answer}\n"
                    turns.append({"player": player_name, "answer": model_answer, "system_prompt": model_out[0]["generated_text"][0]["content"], "turn_prompt": model_out[0]["generated_text"][1]["content"]})
                    if "STOP" in model_answer:
                        conv_round = 4
                        turn = 5
                    else: 
                        turn += 1
                conv_round += 1
            
            # TODO auch prompt speichern
            batch_convs.append(turns)

        return {"output": batch_convs}
                

    def prompt_models(self, split = "mini", device = "cpu"):
        model_id = "meta-llama/Llama-3.2-1B-Instruct"
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
    
        roc_data = ROCStories(split = split)
        prompt_data = roc_data.shuffled_dataset
        outputs = prompt_data.map(lambda batch: self.play_games(batch, pipe), batched=True)

        model_name = model_id.split("/")[-1]
        os.makedirs(f"./outputs/multi/{model_name}/{split}", exist_ok = True)
        outfile = f"./outputs/multi/{model_name}/{split}/outputs_multi_{model_name}_{split}_{str(datetime.now()).replace(" ", "_")}.jsonl"
        outputs.to_json(outfile, indent=4)

        return outfile