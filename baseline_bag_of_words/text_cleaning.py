import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

class TextCleaner():
    def __init__(self) -> None:
        self.lemmatizer = WordNetLemmatizer()
        self.lemmatizer_pos_conversion = {"N":"n","V":"v", "J":"a", "R":"r"}
        self.stop_words = set(stopwords.words('english')) 
        self.stop_words = self.stop_words.union({".",",",":",";","?","!","#","\"","\'","(",")","@card@","I","-","\'s","n\'t","","...","[","]","@ord@","There","|","To","/",""})
        self.unwanted_pos_tags = ['$', "''", '(', ')', ':','IN','LS','MD','PP','TO','VBD','VBN','WDT','WP$','WRB','``','DT','CC','WP']
        self.POS_conversion_list = {'VB':'V','VBZ':'V','NNS':'N', 'JJ':'J', 'NN':'N', 'NP':'N', 'VVD':'V', 'VBP':'V', 'VVP':'V', 'CD':'N', 'RB':'R', 'VVN':'V', 'VVZ':'V', 'PDT':'N', 'VVG':'V', 'VV':'V', 'JJS':'J', 'JJR':'J', 'RBS':'R', 'RBR':'R', 'RP':'R', 'NPS':'N', 'SYM':'N', 'FW':'N', 'UH':'N'}
    
    def clean_word(self, word, pos):
        lemmatizer_pos = self.lemmatizer_pos_conversion[pos]
        lemmatized_word = self.lemmatizer.lemmatize(word, pos=lemmatizer_pos)
        word_pos = lemmatized_word.lower() + "." + pos
        return word_pos
    
    def clean_context(self, context, target_id):
        context = context.split(" ")
        context_tagged = nltk.pos_tag(context)
        cleaned_context = []
        for i_word in range(len(context)):
            if i_word!=target_id:
                current_word_lemmatized = self.lemmatizer.lemmatize(context[i_word]).lower()
                current_word_pos = context_tagged[i_word][1]
                if (current_word_lemmatized not in self.stop_words) and (current_word_pos not in self.unwanted_pos_tags):
                    print(current_word_lemmatized)
                    proper_pos_tag = self.POS_conversion_list[current_word_pos]
                    cleaned_context.append(current_word_lemmatized + "." + proper_pos_tag)
        
        return cleaned_context
    