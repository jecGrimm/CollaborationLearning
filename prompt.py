import torch
from transformers import pipeline
from data import ROCStories
import os
import time
from datetime import datetime
from tqdm import tqdm

class Agent:
    def __init__(self, model_id, split, framework):
        self.model_id = model_id
        self.model_name = model_id.split("/")[-1]
        self.split = split
        self.framework = framework
        self.date = str(datetime.now()).replace(" ", "_")


    def save_outputs(self, outputs, shard_idx=None):
        outfile = self.define_outfile(shard_idx=shard_idx)
        outputs.to_json(outfile, indent=4)


    def define_outfile(self, shard_idx = None):
        if shard_idx == None:
            os.makedirs(f"./outputs/{self.framework}/{self.model_name}/{self.split}/", exist_ok = True)
            return f"./outputs/{self.framework}/{self.model_name}/{self.split}/outputs_{self.framework}_{self.model_name}_{self.split}_{self.date}.jsonl"
        else:
            os.makedirs(f"./outputs/{self.framework}/{self.model_name}/{self.split}/ckps/", exist_ok = True)
            return f"./outputs/{self.framework}/{self.model_name}/{self.split}/ckps/outputs_{self.framework}_{self.model_name}_{self.split}_{self.date}_{shard_idx}.jsonl"

class SingleAgent(Agent):
    def __init__(self, model_id, split):
        self.model_id = model_id
        self.split = split

        super().__init__(model_id=model_id, split=split, framework="single")

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
    
    def prompt_model(self, logger, device = "cpu", batch_size = 1):
        model_id = "meta-llama/Llama-3.2-1B-Instruct"
        pipe = pipeline(
            "text-generation",
            model=model_id,
            dtype=torch.bfloat16,
            device_map=device
        )
        pipe.model.generation_config.pad_token_id = pipe.model.config.eos_token_id[0]
        #pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id[0]
        pipe.tokenizer.padding_side = "left"
        logger.log_pipe(pipe)
    
        roc_data = ROCStories(split = self.split)
        logger.run_data.update({"shuffle_file": roc_data.shuffle_file, "batch_size": batch_size})

        prompt_data = roc_data.shuffled_dataset.map(self.create_message, batched=True)
        full_output_list = []

        start = time.time()
        # process data in smaller parts for checkpointing
        num_shards = int((len(prompt_data)/1000))+1
        processed_inst = 0 # track number of instances
        for shard_idx in tqdm(range(num_shards), desc = "Processing shards", total = num_shards):
            # process shard
            shard_data = prompt_data.shard(num_shards=num_shards, index=shard_idx)
            outputs = tqdm([out for out in pipe(
                    list(shard_data["prompt"]),
                    max_new_tokens=256,
                    batch_size = batch_size
                )], desc = "Processing batches", total = len(list(shard_data["prompt"])))
            
            # store shard output
            output_list = [turn["generated_text"].split("Answer:\n")[-1] for conv in outputs for turn in conv]
            full_output_list += output_list

            # TODO: store all current instances in ckp-file and not only the shard
            output_data = shard_data.add_column("output", output_list)
            
            self.save_outputs(outputs=output_data, shard_idx=shard_idx)

            processed_inst += len(shard_data)
            logger.run_data.update({"instances": processed_inst})

            logger.store_run_data()


        end = time.time()
        logger.run_data.update({"duration": end - start}) # track processing time

        #print(outputs[0][0]["generated_text"])

        # output_list = [turn["generated_text"] for conv in outputs for turn in conv]
        # num_missing = len(prompt_data) - len(output_list)
        # output_list = output_list + [""] * num_missing

        # save full output
        output_data = prompt_data.add_column("output", full_output_list)
        logger.run_data.update({"instances": len(full_output_list)})

        # model_name = model_id.split("/")[-1]
        # os.makedirs(f"./outputs/single/{model_name}/{split}", exist_ok = True)
        # outfile = f"./outputs/single/{model_name}/{split}/outputs_single_{model_name}_{split}_{str(datetime.now()).replace(" ", "_")}.jsonl"
        # output_data.to_json(outfile, indent=4)
        self.save_outputs(output_data)

class MultiAgent(Agent):
    def __init__(self, model_id, split):
        super().__init__(model_id=model_id, split=split, framework="multi")

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

    def play_games(self, batch, pipe, batch_size):
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
                            batch_size = batch_size
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
            
            batch_convs.append(turns)

        return {"output": batch_convs}
                
    def prompt_models(self, logger, device = "cpu", batch_size = 1):
        pipe = pipeline(
            "text-generation",
            model=self.model_id,
            dtype=torch.bfloat16,
            device_map=device,
        )
        logger.log_pipe(pipe)
    
        roc_data = ROCStories(split = self.split)
        logger.run_data.update({"shuffle_file": roc_data.shuffle_file, "batch_size": batch_size})

        prompt_data = roc_data.shuffled_dataset
        full_output_list = []

        start = time.time()

        num_shards = int((len(prompt_data)/1000))+1
        processed_inst = 0
        for shard_idx in tqdm(range(num_shards), desc = "Processing shards"):
            shard_data = prompt_data.shard(num_shards=num_shards, index=shard_idx)
            outputs = shard_data.map(lambda batch: self.play_games(batch, pipe, batch_size), batched=True, batch_size=batch_size)
            self.save_outputs(outputs, shard_idx=shard_idx)

            full_output_list += outputs["output"]

            processed_inst += len(prompt_data)
            logger.run_data.update({"instances": processed_inst})
            
            logger.store_run_data()

        end = time.time()
        logger.run_data.update({"duration": end - start})

        output_data = prompt_data.add_column("output", full_output_list)
        self.save_outputs(output_data)