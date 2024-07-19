import numpy as np

gold_file = open("dataset/LS14/test_refactored/coinco_test_multitokens_singleword.gold",'r').read().split("\n")[:-1]
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
    


result_file = open("dataset/results/coinco_results_gapmultitoken_singleword_PURE-GLOSS_6809_probabilites.txt",'r').read().split("\n")[:-1]
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



from nltk.stem import WordNetLemmatizer
 
lemmatizer = WordNetLemmatizer()
 
print("rocks :", lemmatizer.lemmatize("rocks"))

N = len(gold_dict)
best_total = []
best_mode_total = []
for id in list(gold_dict.keys()):
    gold_substitutes = gold_dict[id]["substitutes"]
    best_guess = result_dict[id][0]
    if best_guess in [lemmatizer.lemmatize(w) for w in list(gold_substitutes.keys())]:
        best_total.append((gold_substitutes[best_guess])/sum(list(gold_substitutes.values())))
    else:
        best_total.append(0.)
    if best_guess == lemmatizer.lemmatize(list(gold_substitutes.keys())[0]):
        best_mode_total.append(1.)
    else:
        best_mode_total.append(0.)
print(f"best measure : {100*sum(best_total)/N:.02f} %, mode : {100*sum(best_mode_total)/N:.02f} %")

oot_total = 0.
oot_mode_total = 0.
for id in list(gold_dict.keys()):
    gold_substitutes = gold_dict[id]["substitutes"]
    oot_guess_witness = False
    for guess in result_dict[id]:
        guess = lemmatizer.lemmatize(guess)
        if guess in [lemmatizer.lemmatize(w) for w in list(gold_substitutes.keys())]:
            oot_total = oot_total + (gold_substitutes[guess])/sum(list(gold_substitutes.values()))
        if guess == lemmatizer.lemmatize(list(gold_substitutes.keys())[0]) and (not oot_guess_witness):
            oot_guess_witness = True
            oot_mode_total += 1.
print(f"oot measure : {100*oot_total/N:.02f} %, mode : {100*oot_mode_total/N:.02f} %")
