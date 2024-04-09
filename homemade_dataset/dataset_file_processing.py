import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
#
parser.add_argument("-uf", "--unprocessed_file_fn", type=str, help="path of unprocessed dataset file",
                    default="/home/user/Documents/KULeuven/Master Thesis/lexsubcon-mod-test/homemade_dataset/unprocessed_dataset_file.txt")

# --------------- train dataset
parser.add_argument("-pf", "--preprocessed_file_to_create_fn", type=str, help="path of procecessed dataset file we are gonna create",
                    default='/home/user/Documents/KULeuven/Master Thesis/lexsubcon-mod-test/homemade_dataset/homemade_dataset.preprocessed')
parser.add_argument("-gf", "--golden_file_to_create_fn", type=str, help="path of gold of dataset file we are gonna create",
                    default='/home/user/Documents/KULeuven/Master Thesis/lexsubcon-mod-test/homemade_dataset/homemade_dataset.gold')

args = parser.parse_args()



def detect_and_split(sentence_part, cleaned_split_sentence_in_progress, character="."):
    assert len(character)==1
    
    if character in sentence_part:
        if sentence_part[0]==character:
            #print(f"\" {character} \" at the start of sentence part, splitting")
            cleaned_split_sentence_in_progress.append(character)
            cleaned_split_sentence_in_progress.append(sentence_part[1:])
        elif sentence_part[-1]==character:
            #print(f"\" {character} \" at the end of sentence part, splitting")
            cleaned_split_sentence_in_progress.append(sentence_part[:-1])
            cleaned_split_sentence_in_progress.append(character)
        else:
            print(f"\" {character} \" found in middle of sentence part. Cannot handle that. Exiting.")
            print(sentence_part)
            exit(0)
    else:
        cleaned_split_sentence_in_progress.append(sentence_part)
    return cleaned_split_sentence_in_progress


def clean_sentence(sentence):
    split_sentence = sentence.split(" ")
    current_pass_cleaned_split_sentence = split_sentence
    temp_next_pass_split_sentence = []
    for i in range(len(current_pass_cleaned_split_sentence)):
        current_part = current_pass_cleaned_split_sentence[i].lower()
        temp_next_pass_split_sentence = detect_and_split(current_part, temp_next_pass_split_sentence, character=".")
    
    current_pass_cleaned_split_sentence = temp_next_pass_split_sentence
    temp_next_pass_split_sentence = []
    for i in range(len(current_pass_cleaned_split_sentence)):
        current_part = current_pass_cleaned_split_sentence[i].lower()
        temp_next_pass_split_sentence = detect_and_split(current_part, temp_next_pass_split_sentence, character=",")
    
    current_pass_cleaned_split_sentence = temp_next_pass_split_sentence
    temp_next_pass_split_sentence = []
    for i in range(len(current_pass_cleaned_split_sentence)):
        current_part = current_pass_cleaned_split_sentence[i].lower()
        temp_next_pass_split_sentence = detect_and_split(current_part, temp_next_pass_split_sentence, character=";")
    
    current_pass_cleaned_split_sentence = temp_next_pass_split_sentence
    temp_next_pass_split_sentence = []
    for i in range(len(current_pass_cleaned_split_sentence)):
        current_part = current_pass_cleaned_split_sentence[i].lower()
        temp_next_pass_split_sentence = detect_and_split(current_part, temp_next_pass_split_sentence, character=":")
    
    first_pass_cleaned_split_sentence = temp_next_pass_split_sentence
    second_pass_cleaned_sentence = []
    for i in range(len(first_pass_cleaned_split_sentence)):
        current_part = first_pass_cleaned_split_sentence[i]
        if "\'s" in current_part:
            #print("\" \'s \" found in sentence part. splitting")
            current_part_split = current_part.split("\'")
            second_pass_cleaned_sentence.append(current_part_split[0])
            second_pass_cleaned_sentence.append("\'" + current_part_split[1])
        else:
            second_pass_cleaned_sentence.append(current_part)
    
    third_pass_cleaned_sentence = []
    for i in range(len(second_pass_cleaned_sentence)):
        current_part = second_pass_cleaned_sentence[i]
        if "n\'t" in current_part:
            #print("\" n\'t \" found in sentence part. splitting")
            #print(current_part)
            current_part_split = current_part.split("n\'t")
            third_pass_cleaned_sentence.append(current_part[:-3])
            third_pass_cleaned_sentence.append(current_part[-3:])
        else:
            third_pass_cleaned_sentence.append(current_part)
    #print(third_pass_cleaned_sentence)
    cleaned_sentence = " ".join(third_pass_cleaned_sentence)
    #print(cleaned_sentence)
    for i in range(len(third_pass_cleaned_sentence)):
        current_part = third_pass_cleaned_sentence[i]
        if "\'" in current_part:
            print("\" \' \" found in sentence part. showing")
            print(current_part)
    return cleaned_sentence
    
def locate_target_index_in_context(target_word, context):
    found_target_index = -1
    split_context = context.split(" ")
    for i in range(len(split_context)):
        if split_context[i] == target_word:
            return i
    print("target word not found in context. indicate index.")
    print(target_word)
    displayable_context = {}
    for i in range(len(split_context)):
        displayable_context[i]=split_context[i]
    print(displayable_context)
    return int(input("index : "))

"""
dataset contains 90 'N', 90 'V', 25 'J'

Format of the returned object: list([
    dict({target:"word.N",
    context:"the context sentence unprocessed",
    target_index:42
    substitutes:list(["gold_substitute_1","gold_substitute_2", ...])}),
    
    dict({...}),
    ...
])
"""
def read_unprocessed_file_and_put_in_list_dict_format(unprocessed_file_filename):
    data_list = open(unprocessed_file_filename,'r').read().split("\n")
    data_list_clean = []
    for i in range(len(data_list)):
        if data_list[i] != "":
            data_list_clean.append(data_list[i])
    target_words = data_list_clean[0::3]
    target_poss = []
    for i in range(len(target_words)):
        current = target_words[i].split(".")
        current_pos = current[1]
        current_word = current[0]
        target_poss.append(current_pos)
        target_words[i] = current_word
    contexts = data_list_clean[1::3]
    for i in range(len(contexts)):
        contexts[i] = clean_sentence(contexts[i])
    substitutes = data_list_clean[2::3]
    for i in range(len(substitutes)):
        substitutes[i] = substitutes[i].lower()
        substitutes[i] = substitutes[i].split(", ")
        #print(substitutes[i])
        
    target_indexes = [20, 4, 2, 12, 2, 1, 6, 9, 5, 3, 4, 3, 5, 6, 14, 14, 1, 2, 3, 4, 3, 5, 5, 3, 2, 7, 13, 8, 1, 5, 2, 3, 3, 1, 5, 4, 2, 5, 1, 1, 9, 4, 1, 1, 8, 6, 1, 6, 4, 1, 11, 7, 5, 3, 5, 7, 2, 3, 3, 5, 1, 4, 12, 9, 4, 3, 4, 3, 4, 3, 8, 7, 18, 21, 2, 8, 4, 3, 4, 4, 2, 4, 1, 4, 10, 7, 3, 2, 3, 2, 5, 2, 10, 3, 4, 5, 10, 10, 5, 3, 11, 9, 7, 17, 3, 5, 6, 11, 3, 3, 4, 1, 5, 12, 10, 2, 5, 2, 2, 7, 2, 3, 2, 7, 2, 4, 3, 9, 1, 11, 4, 7, 2, 4, 3, 6, 8, 4, 1, 10, 6, 3, 7, 7, 5, 1, 19, 3, 3, 8, 4, 7, 2, 7, 2, 3, 2, 3, 2, 6, 3, 3, 3, 4, 2, 3, 3, 4, 6, 4, 4, 3, 14, 7, 1, 2, 3, 3, 9, 4, 4, 6, 5, 7, 1, 7, 5, 2, 3, 3, 3, 3, 4, 9, 3, 3, 2, 3, 4, 1, 3, 3, 3, 1]
    """
    target_indexes = []
    for i in range(len(target_words)):
        target_indexes.append(locate_target_index_in_context(target_word=target_words[i], context=contexts[i]))
    print(target_indexes)
    input('next')
    """
    dataset_list_dicts = []
    for i in range(len(target_words)):
        current_dict = {"target":target_words[i]+"."+target_poss[i],
                        "context":contexts[i],
                        "target_index":target_indexes[i],
                        "substitutes":substitutes[i]}
        dataset_list_dicts.append(current_dict)
    return dataset_list_dicts

def write_preprocessed_file(preprocessed_file_fn, dataset_list_dict):
    file = open(preprocessed_file_fn, 'w')
    for i in range(len(dataset_list_dict)):
        current_dict = dataset_list_dict[i]
        line = ""
        line += current_dict["target"] + "\t"
        line += str(i+1) + "\t"
        line += str(current_dict["target_index"]) + "\t"
        line += current_dict["context"]+"\n"
        file.write(line)
    file.close()
    
def write_gold_file(gold_file_fn, dataset_list_dict):
    file = open(gold_file_fn, 'w')
    for i in range(len(dataset_list_dict)):
        current_dict = dataset_list_dict[i]
        line = ""
        line += current_dict["target"] + " "
        line += str(i+1) + " ::  "
        for j in range(len(current_dict["substitutes"])):
            line += current_dict["substitutes"][j] + " 1;"
        line += "\n"
        file.write(line)
    file.close()

dataset_list_dict = read_unprocessed_file_and_put_in_list_dict_format(unprocessed_file_filename = args.unprocessed_file_fn)
write_preprocessed_file(preprocessed_file_fn=args.preprocessed_file_to_create_fn, dataset_list_dict=dataset_list_dict)
write_gold_file(gold_file_fn=args.golden_file_to_create_fn, dataset_list_dict=dataset_list_dict)