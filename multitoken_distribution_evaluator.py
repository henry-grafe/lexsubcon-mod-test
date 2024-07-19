from reader import Reader_lexical
import argparse
from tqdm import tqdm
from src.transformers import BertTokenizer
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-tt", "--test_file", type=str, help="path of the test file dataset",
                    default='dataset/LS14/test_refactored/coinco_test_multitokens.preprocessed')
args = parser.parse_args()
reader = Reader_lexical()
reader.create_feature(args.test_file)

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased", do_lower_case=False)
lengths_tokens=[]
lengths_words=[]

for main_word in tqdm(reader.words_candidate):
        for instance in reader.words_candidate[main_word]:
            for context in reader.words_candidate[main_word][instance]:
                #print(context)
                #print(main_word)
                #input("next")
                change_word = context[0]
                main_word_without_pos = main_word.split(".")[0]
                lengths_words.append(len(main_word_without_pos.split(" ")))
                if lengths_words[-1]>1:
                    print(main_word_without_pos)
                    input('next')
                text = context[1]
                original_text = text
                index_word = context[2]
                change_word = text.split(' ')[int(index_word)]
                synonyms = []
                word_tokens = tokenizer.tokenize(change_word)
                lengths_tokens.append(len(word_tokens))
                

lengths_tokens=np.array(lengths_tokens,dtype="int")
print(lengths_tokens.min(), lengths_tokens.max())
n_tokens=np.arange(2,8)
counts=np.zeros(n_tokens.shape,dtype="int")
for i in range(len(n_tokens)):
    counts[i] = (lengths_tokens==n_tokens[i]).sum()
print(n_tokens)
print(counts)
n_tokens=[str(n_tokens[i]) for i in range(len(n_tokens))]

plt.bar(n_tokens,counts)
plt.xlabel("Number of tokens")
plt.ylabel("Count")
plt.title("Distribution of sizes of words in the set")
plt.show()


lengths_words=np.array(lengths_words,dtype="int")
print(lengths_words.min(), lengths_words.max())
n_words=np.arange(1,8)
counts=np.zeros(n_words.shape,dtype="int")
for i in range(len(n_words)):
    counts[i] = (lengths_words==n_words[i]).sum()
print(n_words)
print(counts)
n_words=[str(n_words[i]) for i in range(len(n_words))]

plt.bar(n_words,counts)
plt.xlabel("Number of words")
plt.ylabel("Count")
plt.title("Distribution of number of words in the set")
plt.show()