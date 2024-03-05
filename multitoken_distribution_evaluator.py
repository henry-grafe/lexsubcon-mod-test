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
lengths=[]

for main_word in tqdm(reader.words_candidate):
        for instance in reader.words_candidate[main_word]:
            for context in reader.words_candidate[main_word][instance]:
                #print(context)
                #print(main_word)
                #input("next")
                change_word = context[0]
                text = context[1]
                original_text = text
                index_word = context[2]
                change_word = text.split(' ')[int(index_word)]
                synonyms = []
                word_tokens = tokenizer.tokenize(change_word)
                lengths.append(len(word_tokens))

lengths=np.array(lengths,dtype="int")
print(lengths.min(), lengths.max())
n=np.arange(2,8)
counts=np.zeros(n.shape,dtype="int")
for i in range(len(n)):
    counts[i] = (lengths==n[i]).sum()
print(n)
print(counts)
n=[str(n[i]) for i in range(len(n))]

plt.bar(n,counts)
plt.xlabel("Number of tokens")
plt.ylabel("Count")
plt.title("Distribution of sizes of words in the set")
plt.show()