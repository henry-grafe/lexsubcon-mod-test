import pickle
import numpy as np
from copy import deepcopy
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from . import vector_operations
from . import text_cleaning
stop_words = set(stopwords.words('english')) 
stop_words = stop_words.union({".",",",":",";","?","!","#","\"","\'","(",")","@card@","I","-","\'s","n\'t","","...","[","]","@ord@","There","|","To","/",""})
unwanted_pos_tags = ['$', "''", '(', ')', ':','IN','LS','MD','PP','TO','VBD','VBN','WDT','WP$','WRB','``','DT','CC','WP']
POS_conversion_list = {'VBG':'V','NNS':'N', 'JJ':'J', 'NN':'N', 'NP':'N', 'VVD':'V', 'VVP':'V', 'CD':'N', 'RB':'R', 'VVN':'V', 'VVZ':'V', 'PDT':'N', 'VVG':'V', 'VV':'V', 'JJS':'J', 'JJR':'J', 'RBS':'R', 'RBR':'R', 'RP':'R', 'NPS':'N', 'SYM':'N', 'FW':'N', 'UH':'N'}
class CorpusRetriever():
    def __init__(self, cleaned_corpus_fp) -> None:
        self.texts = pickle.load(open(cleaned_corpus_fp,'rb'))
        self.lemmatizer = WordNetLemmatizer()
        
    def find_related_sentences(self, word_pos):
        related_sentences = []
        for i_text in range(len(self.texts)):
            for j_sentences in range(len(self.texts[i_text])):
                if word_pos in self.texts[i_text][j_sentences]:
                    related_sentences.append(self.texts[i_text][j_sentences])
        
        for i_sentence in range(len(related_sentences)):
            new_sentence = []
            for j_word in range(len(related_sentences[i_sentence])):
                if related_sentences[i_sentence][j_word] != word_pos:
                    new_sentence.append(related_sentences[i_sentence][j_word])
            related_sentences[i_sentence] = new_sentence
        
        return related_sentences
    
    def find_related_sentences_of_multiple_words(self, word_pos_list):
        related_sentences = []
        for k_word_pos in range(len(word_pos_list)):
            related_sentences.append([])
            
        for i_text in range(len(self.texts)):
            for j_sentences in range(len(self.texts[i_text])):
                for k_word_pos in range(len(word_pos_list)):
                    if word_pos_list[k_word_pos] in self.texts[i_text][j_sentences]:
                        related_sentences[k_word_pos].append(self.texts[i_text][j_sentences])
        
        
        for k_word_pos in range(len(word_pos_list)):
            for i_sentence in range(len(related_sentences[k_word_pos])):
                new_sentence = []
                for j_word in range(len(related_sentences[k_word_pos][i_sentence])):
                    if related_sentences[k_word_pos][i_sentence][j_word] != word_pos_list[k_word_pos]:
                        new_sentence.append(related_sentences[k_word_pos][i_sentence][j_word])
                related_sentences[k_word_pos][i_sentence] = new_sentence
        
        return related_sentences