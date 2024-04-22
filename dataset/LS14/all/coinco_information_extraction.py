from bs4 import BeautifulSoup


def find_word_indices_in_context(context, word):
    lowercase_context = context.lower()
    index = 0
    found = False
    while (index <= len(context)-len(word)) and (not found):
        found = True
        for i_relative in range(len(word)):
            if context[index + i_relative] != word[i_relative]:
                found = False
        index += 1
    if not found:
        return (-1, -1)
    else:
        return (index-1, index-1+len(word))
    
    

data = open("/home/user/Documents/KULeuven/Master Thesis/lexsubcon-mod-test/dataset/LS14/all/coinco.xml",'r', encoding='utf-8').read()
bs_data = BeautifulSoup(data, 'xml')
del data
sents = bs_data.find_all("sent")

dataset = {}

for i_sent in range(len(sents)):
    sent = sents[i_sent]
    context = sent.find("targetsentence").text[5:-5]
    print(context)
    tokens = sent.find_all("token")
    for j_token in range(len(tokens)):
        token = tokens[j_token]
        id = token['id']
        if id != 'XXX':
            id = int(id)
            dataset[id] = {"context":context, "word":token['wordform'], "lemma":token["lemma"]}
            start_index, end_index = find_word_indices_in_context(context, dataset[id]["word"])
            dataset[id]["indices"] = [start_index, end_index]
            print(dataset[id])
    input("next")
