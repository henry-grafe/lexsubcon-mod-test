def get_gold_dict(filename):
    gold_file = open(filename,'r').read().split("\n")[:-1]
    gold_dict = {}
    for line in gold_file:
        line = line.split("::")
        line[1] = line[1][2:]
        line[0] = line[0][:-1].split(" ")
        
        id = int(line[0][-1])
        
        target_word = line[0][0].split(".")[0]
        target_pos = line[0][0].split(".")[-1]
        gold_substitutes_line = line[1].split(";")[:-1]
        
        if id == 7611:
            gold_substitutes_line = gold_substitutes_line[:-2]
        
        gold_substitutes_line_dict = {element[:-2]:int(element[-1]) for element in gold_substitutes_line}
        gold_dict[id] = {"target_word":target_word, "target_pos":target_pos, "substitutes":gold_substitutes_line_dict}
    return gold_dict
    

def get_result_dict(filename):
    result_file = open(filename,'r').read().split("\n")[:-1]
    result_dict = {}
    for line in result_file:
        line = line.split('\t')
        id = int(line[1].split(" ")[-1])
        results_substitutes = []
        for i in range(2, 2+10):
            splitted = line[i].split(" ")
            word = " ".join(splitted[:-1])
            results_substitutes.append(word)
        result_dict[id] = results_substitutes
    return result_dict

gold_dict = get_gold_dict("homemade_dataset/homemade_dataset.gold")

total_substitute = 0
multi_word_substitues = 0
for id in list(gold_dict.keys()):
    substitutes = gold_dict[id]["substitutes"]
    for substitute in list(substitutes.keys()):
        total_substitute += 1
        if len(substitute.split(" ")) > 1:
            #print(substitute)
            multi_word_substitues += 1
        else:
            print(substitute, id)

print(total_substitute, multi_word_substitues, multi_word_substitues/total_substitute)

total_instances = 0
instances_with_at_least_one_multiword_gold_substitutes = 0
for id in list(gold_dict.keys()):
    total_instances += 1
    multi_word_substitute_present = False
    substitutes = gold_dict[id]["substitutes"]
    for substitute in list(substitutes.keys()):
        if len(substitute.split(" ")) > 1:
            #print(substitute)
            multi_word_substitute_present = True
    if multi_word_substitute_present:
        instances_with_at_least_one_multiword_gold_substitutes += 1

print(total_instances, instances_with_at_least_one_multiword_gold_substitutes, instances_with_at_least_one_multiword_gold_substitutes/total_instances)

all_substitutes_scores_array = []
multi_word_substitutes_scores_array = []
for id in list(gold_dict.keys()):
    substitutes = gold_dict[id]["substitutes"]
    for substitute in list(substitutes.keys()):
        all_substitutes_scores_array.append(substitutes[substitute])
        if len(substitute.split(" ")) > 1:
            multi_word_substitutes_scores_array.append(substitutes[substitute])

import matplotlib.pyplot as plt
import numpy as np

all_substitutes_scores_array = np.array(all_substitutes_scores_array)
multi_word_substitutes_scores_array = np.array(multi_word_substitutes_scores_array)

print(all_substitutes_scores_array.mean())
print(multi_word_substitutes_scores_array.mean())

plt.subplot(1,2,1)
unique_scores_all, counts_all = np.unique(all_substitutes_scores_array, return_counts=True)
plt.bar(unique_scores_all, counts_all/counts_all.sum())
plt.title("Score Distribution of the Ground Truth Substitutes, All")
plt.xlabel("Ground Truth Score")
plt.ylabel("Density")
#plt.subplot(1,2,2)
unique_scores_multiwords, counts_multiwords = np.unique(multi_word_substitutes_scores_array, return_counts=True)
unique_scores_multiwords=np.append(unique_scores_multiwords,np.array([8,9]))
counts_multiwords=np.append(counts_multiwords,np.array([0,0]))
plt.bar(unique_scores_multiwords, counts_multiwords/counts_multiwords.sum(),alpha=0.8)
plt.xlabel("Ground Truth Score")
plt.ylabel("Density")
plt.title("Score Distribution of the Ground Truth Substitutes, Multi-Word only")

plt.show()

