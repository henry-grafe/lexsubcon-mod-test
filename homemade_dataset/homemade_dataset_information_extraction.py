import json

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

def create_homemade_dataset_dict(filename="/home/user/Documents/KULeuven/Master Thesis/lexsubcon-mod-test/homemade_dataset/homemade_dataset.preprocessed"):
    """
    data = open(filename,'r', encoding='utf-8').read().split("\n")[:-1]
    unprocessed_data = open("/home/user/Documents/KULeuven/Master Thesis/lexsubcon-mod-test/homemade_dataset/unprocessed_dataset_file.txt",'r',encoding='utf-8').read().split('\n')
    unprocessed_sentences = unprocessed_data[1::4]
    print(unprocessed_sentences)
    print(len(unprocessed_sentences))
    dataset={}
    for i_line in range(len(data)):
        line = data[i_line].split("\t")
        word = line[0].split(".")[0]
        context = line[3]
        id = int(line[1])
        start_index, end_index = find_word_indices_in_context(unprocessed_sentences[id-1], word)
        
        s = ""
        for i in range(len(unprocessed_sentences[id-1])):
            s+=str(i)+unprocessed_sentences[id-1][i]
        print(word)
        print(unprocessed_sentences[id-1])
        print(unprocessed_sentences[id-1][start_index:end_index])
        
        print(s)
        change_flag = input("change ? : ").lower()
        if change_flag=="y":
            start_index = int(input("start index : "))
            end_index = int(input("end index : "))
        
        dataset[id] = {"context":context, "word":word, "indices":[start_index, end_index]}
    
    with open('/home/user/Documents/KULeuven/Master Thesis/lexsubcon-mod-test/homemade_dataset/homemade_dataset.json', 'w') as f:
        json.dump(dataset, f)
    """
    return_dict =  json.load(open('/home/user/Documents/KULeuven/Master Thesis/lexsubcon-mod-test/homemade_dataset/homemade_dataset.json','r'))
    data = open(filename,'r', encoding='utf-8').read().split("\n")[:-1]
    unprocessed_data = open("/home/user/Documents/KULeuven/Master Thesis/lexsubcon-mod-test/homemade_dataset/unprocessed_dataset_file.txt",'r',encoding='utf-8').read().split('\n')
    unprocessed_sentences = unprocessed_data[1::4]
    
    for i_line in range(len(data)):
        line = data[i_line].split("\t")
        word = line[0].split(".")[0]
        
        id = int(line[1])
        context = unprocessed_sentences[id-1]
        return_dict[str(id)]["context"] = context
    
    return {int(key):value for key, value in return_dict.items()}
    