from datasets import load_dataset
import random

class ROCStories:
    def __init__(self):
        self.orig_dataset = load_dataset("csv", data_files="data/ROCStories_winter2017 - ROCStories_winter2017.csv", split="train")
        self.shuffled_dataset = self.orig_dataset.map(self.shuffle_sents, batched = True)

    def shuffle_sents(self, batch):
        batch_shuffled = list()
        for inst in batch:
            ordered_sents = [batch["sentence1"], batch["sentence2"], batch["sentence3"], batch["sentence4"]]
            batch_shuffled.append(random.shuffle(ordered_sents))

        return {"shuffled_sents": batch_shuffled}
        
if __name__ == "__main__":
    roc_data = ROCStories()
    print(roc_data.shuffled_dataset[:10])