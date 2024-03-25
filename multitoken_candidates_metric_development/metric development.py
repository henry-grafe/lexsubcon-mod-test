import numpy as np
import random
import torch
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


word_list = list(words.words())
def obtain_gold_substitutes_list(filename):
    test_gold = open(filename,'r',encoding="latin5").read().split("\n")[:-1]
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

    test_gold_ids = np.array(test_gold_ids)[test_gold_reording_indexes]
    for i in range(len(test_gold_reordered)):
        test_gold_reordered[i] = [test_gold_ids[i], test_gold_reordered[i]]
    
    return test_gold_reordered
    
def obtain_generated_ranking_list(filename):
    generated_ranking =  open(filename,"r").read().split("\n")[:-1]
    #print(generated_ranking[:10])
    #print(generated_ranking[-10:])
    for i_line in range(len(generated_ranking)):
        generated_ranking[i_line] = generated_ranking[i_line].split("\t")
        #print(generated_ranking[i_line])
        id = int(generated_ranking[i_line][1].split(' ')[-1])
        #print(id)
        substitutes = generated_ranking[i_line][2:]
        new_substitutes = []
        for j_substitute in range(len(substitutes)):
            temp = " ".join(substitutes[j_substitute].split(" ")[:-1])
            new_substitutes.append(temp)
            #print(substitutes[j_substitute])
        generated_ranking[i_line] = [id, new_substitutes]
    return generated_ranking
   
def strip_gold_list(gold_list, word_numbers_to_keep=[2,3,4,5,6]):
    stripped_gold_list = []
    for i_example in range(len(gold_list)):
        id = gold_list[i_example][0]
        substitutes = gold_list[i_example][1]
        stripped_substitutes = []
        for j_substitute in range(len(substitutes)):
            substitute_word = substitutes[j_substitute][0]
            if len(substitute_word.split(" ")) in word_numbers_to_keep:
                stripped_substitutes.append(substitutes[j_substitute])
        if len(stripped_substitutes) != 0:
            stripped_gold_list.append([id, stripped_substitutes])
    return stripped_gold_list

def strip_generated_list(generated_list, word_numbers_to_keep=[2,3,4,5,6]):
    stripped_generated_list = []
    for i_example in range(len(generated_list)):
        id = generated_list[i_example][0]

        substitutes = generated_list[i_example][1]
        stripped_substitutes = []
        for j_substitute in range(len(substitutes)):
            substitute_word = substitutes[j_substitute]
            if len(substitute_word.split(" ")) in word_numbers_to_keep:
                stripped_substitutes.append(substitutes[j_substitute])
        if len(stripped_substitutes) != 0:
            stripped_generated_list.append([id, stripped_substitutes])
    return stripped_generated_list


def compute_discounted_gain(stripped_gold_list, stripped_generated_list, power=1):
    total = 0
    for i_line in range(len(stripped_gold_list)):
        current_gold_id = stripped_gold_list[i_line][0]

        total_freq = 0
        gold_words = []
        
        for j_substitute in range(len(stripped_gold_list[i_line][1])):
            total_freq += stripped_gold_list[i_line][1][j_substitute][1]
            gold_words.append(stripped_gold_list[i_line][1][j_substitute][0])
        freq_lower_in_ranking = []
        retained_generated_index = -1
        
        for i_line_generated in range(len(stripped_generated_list)):
            id = stripped_generated_list[i_line_generated][0]
            if id == current_gold_id:
                retained_generated_index = i_line_generated
        local_total = 0
        if retained_generated_index != -1:
            for j_rank_position in range(len(stripped_generated_list[retained_generated_index][1])):
                current_substitute = stripped_generated_list[retained_generated_index][1][j_rank_position]
                gold_word_index = -1
                for k_gold in range(len(stripped_gold_list[i_line][1])):
                    if current_substitute == stripped_gold_list[i_line][1][k_gold][0]:
                        #print(f"MATCH found, as gold item {current_gold_id}, in generated item {stripped_generated_list[retained_generated_index][0]} : \"{stripped_gold_list[i_line][1][k_gold][0]}\" at rank position {j_rank_position}")
                        gold_word_index = k_gold
                if gold_word_index != -1:
                    n_higher_freq_before_in_ranking = np.sum(np.array(freq_lower_in_ranking) >= stripped_gold_list[i_line][1][gold_word_index][1])
                    #print(f"for this match, number of elements higher in the ranking with a higher or equal freq : {n_higher_freq_before_in_ranking}")
                    #print(f"factor is thus equal to {1./(j_rank_position+1 - n_higher_freq_before_in_ranking):.02f}")
                    local_total += 1./(j_rank_position+1 - n_higher_freq_before_in_ranking)**power
                    #print(1./(j_rank_position+1 - n_lower_freq_before_in_ranking))
                    freq_lower_in_ranking.append(stripped_gold_list[i_line][1][gold_word_index][1])
        local_total = local_total / float(len(stripped_gold_list[i_line][1]))
        total += local_total
    return total / float(len(stripped_gold_list))


generated_ranking = obtain_generated_ranking_list("coinco_results_multitoken_prune_gloss_pure_val_proposed_6809_probabilites.txt")

gold_list = obtain_gold_substitutes_list("coinco_test.gold")
stripped_gold_list = strip_gold_list(gold_list=gold_list)


stripped_generated_ranking = strip_generated_list(generated_list=generated_ranking)
power = 10**np.linspace(-4,2,100)
value = np.zeros(100)
from tqdm import tqdm
for i in tqdm(range(len(power))):
    value[i] = compute_discounted_gain(stripped_gold_list=stripped_gold_list, stripped_generated_list=stripped_generated_ranking, power=power[i])
import matplotlib.pyplot as plt
plt.loglog(power, value)
plt.show()

