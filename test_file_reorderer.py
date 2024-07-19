import numpy as np
import random
import torch
from proposal_score.score import Cmasked
import matplotlib.pyplot as plt

seed = 6809
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    print("cuda is available")


proposal = Cmasked(max_seq_length=128, do_lower_case=True, pre_trained="bert-base-uncased")


test_examples = open("dataset/LS14/test/coinco_test.preprocessed",'r',encoding="latin5").read().split("\n")[:-1]
test_examples_ids = []
for i in range(len(test_examples)):
    test_examples_ids.append(int(test_examples[i].split("\t")[1]))

test_examples_reording_indexes = np.argsort(test_examples_ids)
test_examples_reordered = []
for i in range(len(test_examples_reording_indexes)):
    test_examples_reordered.append(test_examples[test_examples_reording_indexes[i]])
test_examples_reordered = test_examples_reordered[:37]+test_examples_reordered[38:]
print(len(test_examples_reordered))



test_gold = open("dataset/LS14/test/coinco_test.gold",'r',encoding="latin5").read().split("\n")[:-1]
test_gold_ids = []
for i in range(len(test_gold)):
    test_gold_ids.append(int(test_gold[i].split("::")[0].split(' ')[-2]))

test_gold_reording_indexes = np.argsort(test_gold_ids)
test_gold_reordered = []
for i in range(len(test_gold_reording_indexes)):
    test_gold_reordered.append(test_gold[test_gold_reording_indexes[i]])
print(len(test_gold_reordered))

final_test_examples = []
final_test_gold = []
word_lengths = []
expression_lengths = []
c=0
for i in range(len(test_examples_reordered)):
    line = test_examples_reordered[i]
    target_word = line.split("\t")[0].split(".")[0]
    
    text = line.split("\t")[-1]
    word_index = int(line.split("\t")[-2])
    #print(word_index, text.split(" ")[word_index], text)
    text, target_word_start_index, target_word_end_index, features, _ = proposal.pre_processed_text(text, word_index,
                                                                                             noise_type="GLOSS")
    is_one_token = (target_word_end_index==target_word_start_index)
    is_one_word = (len(target_word.split(" ")) == 1)
    if (not is_one_token) and (not is_one_word):
        c+=1
        num_tokens = target_word_end_index - target_word_start_index + 1
        word_lengths.append(num_tokens)
        expression_lengths.append(len(target_word.split(" ")))
        final_test_gold.append(test_gold_reordered[i])
        final_test_examples.append(test_examples_reordered[i])
print(c)
values, counts = np.unique(word_lengths, return_counts=True)
print(counts)
plt.bar([str(value) for value in values],counts,color="#4e81ff")
plt.title("Distribution of Lengths in Tokens of the Target Words in \nthe Multi-Token Only Dataset")
plt.xlabel("Length in Tokens")
plt.ylabel("Number of Target Words")
plt.show()
values, counts = np.unique(expression_lengths, return_counts=True)
plt.bar([str(value) for value in values],counts,color="#4e81ff")
plt.title("Distribution of Lengths in Words of the Target Expressions in \nthe Multi-Word Only Dataset")
plt.xlabel("Length in Words")
plt.ylabel("Number of Target Expressions")
plt.show()
final_test_examples_file = open("coinco_test_reordered.preprocessed", 'w', encoding="latin5")
final_test_gold_file = open("coinco_test_reordered.gold", 'w', encoding="latin5")
for i in range(len(final_test_examples)):
    final_test_examples_file.write(final_test_examples[i]+"\n")
    final_test_gold_file.write(final_test_gold[i] + "\n")
final_test_examples_file.close()
final_test_gold_file.close()