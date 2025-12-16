    def play_games(self, batch, pipe):
        '''
        @deprecated
        This function processes conversations iteratively.

        @params
            batch: batch of instances
            pipe: pipeline object
        @returns
            dictionary with the conversational turns and player names
        '''
        batch_turns = []
        batch_players = []
        for i in range(len(batch["storyid"])):
            conv_round = 0
            turn = 0
            conversation = ""
            turns = []
            players = []

            roles = dict(zip(self.NAMES, batch["shuffled_sents"][i]))
            while conv_round < self.conversation_rounds:
                while turn < 5:
                    player_name = self.NAMES[turn]

                    model_out = pipe(
                            self.create_message(conv_round, turn, conversation, roles, player_name),
                            max_new_tokens=256
                    )
                    
                    model_answer = model_out[0]["generated_text"][2]["content"]
                
                    conversation += f"\n{player_name}: {model_answer}\n"
                    #turns.append({"player": player_name, "answer": model_answer, "system_prompt": model_out[0]["generated_text"][0]["content"], "turn_prompt": model_out[0]["generated_text"][1]["content"]})
                    turns.append(model_answer)
                    players.append(player_name)
                    if "STOP" in model_answer and conv_round > 0:
                        conv_round = self.conversation_rounds
                        turn = 5
                    else: 
                        turn += 1
                conv_round += 1
            
            #batch_convs.append(turns)
            batch_turns.append(turns)
            batch_players.append(players)

        return {"turns": batch_turns, "players": batch_players}