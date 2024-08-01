from xml.etree import ElementTree as ET
from lxml import etree
from tqdm import tqdm
import pickle
import numpy as np
import os

UKWAC_dir = "C:/Users/NICOLAS/Documents/KULeuven/master_thesis/datasets/UKWAC/"

for i in range(25):
    filepath = os.path.join(UKWAC_dir, "UKWAC-"+str(i+1)+"_processing_ready.xml")
    parser = etree.XMLParser(recover=True,encoding="latin5")
    print("--------------- doing "+"UKWAC-"+str(i+1) + " ---------------")
    print("loading xml...")
    tree = etree.parse(filepath, parser=parser)
    print("loaded !")
    print("getting root...")
    root = tree.getroot()
    print("root obtained !")
    print("gathering passage lengths")
    c=0
    for element in tqdm(root):
        #print(len(element))
        c += len(element)
    print("passage lengths gathered !")
    print(f"{c} sentences in tree")
    print("determining vocabulary...")
    vocab = {}
    vocab_characters = {}
    for passage in tqdm(root):
        for sentence in passage:
            text = sentence.text
            text = text.split("\n")[1:-1]
            for word_trip in text:
                word=word_trip.split("\t")[0].lower()
                if word in vocab:
                    vocab[word]+=1
                else:
                    vocab[word]=1
                for i in range(len(word)):
                    char = str(word[i])
                    if char in vocab_characters:
                        vocab_characters[char]+=1
                    else:
                        vocab_characters[char]=1

    print("vocabulary determined !")
    print(f"vocabulary size is {len(vocab)}")
    print(f"character vocabulary size is {len(vocab_characters)}")

    pickle.dump(obj=vocab,file=open(os.path.join(UKWAC_dir, "UKWAC-"+str(i+1)+"_vocabulary.pickle"),'wb'))
    pickle.dump(obj=vocab_characters,file=open(os.path.join(UKWAC_dir, "UKWAC-"+str(i+1)+"_vocabulary_characters.pickle"),'wb'))