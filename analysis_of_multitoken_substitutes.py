import numpy as np
import random
import torch
from proposal_score.score import Cmasked
from nltk.corpus import words

just_reorder = True
seed = 6809
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

proposal = Cmasked(max_seq_length=128, do_lower_case=True, pre_trained="bert-base-uncased")
proposal.get_possible_words()

word_list = list(words.words())

test_gold = open("dataset/LS14/test/coinco_test.gold",'r',encoding="latin5").read().split("\n")[:-1]
test_gold_ids = []
for i in range(len(test_gold)):
    test_gold_ids.append(int(test_gold[i].split("::")[0].split(' ')[-2]))

test_gold_reording_indexes = np.argsort(test_gold_ids)
test_gold_reordered = []
for i in range(len(test_gold_reording_indexes)):
    test_gold_reordered.append(test_gold[test_gold_reording_indexes[i]])

for i_line in range(len(test_gold_reordered)):
    #print(test_gold_reordered[i_line])
    temp = test_gold_reordered[i_line].split("::")
    substitutes = temp[1][2:].split(";")[:-1]
    temp_subsitutes= []
    for j_substitute in range(len(substitutes)):
        if substitutes[j_substitute] != '':
            temp_subsitutes.append(substitutes[j_substitute])
    substitutes = temp_subsitutes
    for j_substitute in range(len(substitutes)):
        substitutes[j_substitute] = [substitutes[j_substitute][:-2], int(substitutes[j_substitute][-1])]
    test_gold_reordered[i_line] = substitutes
    
multi_words_count = 0
multi_words_substitutes = []
total_count = 0
for i_line in range(len(test_gold_reordered)):
    substitutes = test_gold_reordered[i_line]
    for j_substitute in range(len(substitutes)):
        total_count += 1
        if " " in substitutes[j_substitute][0]:
            multi_words_count += 1
            multi_words_substitutes.append(substitutes[j_substitute][0])

multi_words_substitutes_with_at_least_one_multitoken_word = 0
for multi_words_substitute in multi_words_substitutes:
    tokens = proposal.tokenizer.tokenize(multi_words_substitute)
    no_multitoken = True
    for token in tokens:
        if "##" in token:
            no_multitoken = False
    if no_multitoken is False:
        multi_words_substitutes_with_at_least_one_multitoken_word += 1
        
print(f"CoInCo contains a total of {total_count} substitutes (some might be the same word)")
print(f"Of these, there are {multi_words_count} that are made of multiple words, or {(100*float(multi_words_count)/float(total_count)):.02f}%")
print(f"Of these that are made of multiple words, there are {multi_words_substitutes_with_at_least_one_multitoken_word} contain at least a word that is made of multiple token, or {(100*float(multi_words_substitutes_with_at_least_one_multitoken_word)/float(multi_words_count)):.02f}%")
found =  []
for word in word_list:
    if " " in word:
        found.append(word)
if found==0:
    print("There are no words in the nltk.corpus.words vocabulary that contain spaces, so no word in this list is a multi-word expression")


tokenizer_vocabulary = list(proposal.tokenizer.get_vocab().keys())
non_middle_token_vocab = []
for i_token in range(len(tokenizer_vocabulary)):
    if "##" not in tokenizer_vocabulary[i_token]:
        non_middle_token_vocab.append(tokenizer_vocabulary[i_token])
print(f"The size of the token vocabulary is {len(tokenizer_vocabulary)}")
print(f"Among these, there are {len(non_middle_token_vocab)} that can be the begining of a word")
print(f"Among these {len(non_middle_token_vocab)} that can be the begining of a word, there are {len(proposal.possible_index)} that are words in themselves")