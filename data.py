from datasets import load_dataset
import random

class ROCStories:
    def __init__(self):
        self.orig_dataset = load_dataset("csv", data_files="data/ROCStories_winter2017 - ROCStories_winter2017.csv", split="train")
        self.shuffled_dataset = self.orig_dataset.map(self.shuffle_sents, batched = True)

    def shuffle_sents(self, batch):
        '''
        This method shuffles the first four sentences in the instance. 

        @param
            batch: batch of ROCStory instances
        @returns batch_shuffled: original instance with additional shuffled story.
        '''
        batch_shuffled = list()
        for i in range(len(batch["storyid"])):
            ordered_sents = [batch["sentence1"][i], batch["sentence2"][i], batch["sentence3"][i], batch["sentence4"][i]]
            ordered_sents_copy = ordered_sents.copy()
            random.shuffle(ordered_sents_copy)
            batch_shuffled.append(ordered_sents_copy)

        return {"shuffled_sents":  batch_shuffled}
        
if __name__ == "__main__":
    roc_data = ROCStories()
    print(roc_data.shuffled_dataset[:10])