import gzip
import json
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")

with gzip.open("/home/user/Documents/KULeuven/Master Thesis/swords/assets/parsed/swords-v1.1_test.json.gz", 'r') as f:
    dataset = json.load(f)

print(len(dataset['targets']))
tokens_lengths = {1:0,2:0,3:0,4:0,5:0,5:0,6:0}
for tid, target_dict in dataset['targets'].items():
    target_word = target_dict["target"]
    target_word_tokens = tokenizer.tokenize(target_word)
    #print(target_word, target_word_tokens)
    tokens_lengths[len(target_word_tokens)] += 1
print(tokens_lengths)

