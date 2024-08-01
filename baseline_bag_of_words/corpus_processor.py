import numpy as np
from bs4 import BeautifulSoup
import pickle
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-fp", "--corpus-xml-fp", type=str, help="path of the corpus xml dataset. It has to be corrected so that all the xml is into on root node, which is not the case by default. This means having to pass the file to corpus_xml_modifier.py")
args = parser.parse_args()

data = open(args.corpus_xml_fp,'r', encoding='latin5').read()
bs_data = BeautifulSoup(data, 'xml')
del data
new_filepath = "..\\datasets\\UKWAC\\UKWAC-1.xml\\UKWAC-1_bs_data.pickle"
pickle.dump(bs_data, open(new_filepath,"wb"))
exit(0)
# Create a set of stop words 
stop_words = set(stopwords.words('english')) 
stop_words = stop_words.union({".",",",":",";","?","!","#","\"","\'","(",")","@card@","I","-","\'s","n\'t","","...","[","]","@ord@","There","|","To","/",""})

key_words_to_invert = ['The', 'Introduction', 'Alcohol', 'Asking', 'Experience', 'Comparison', 'Reported', 'Prevalence', 'Types', 
                       'Trends', 'Weekly', 'Estimating', 'Frequency', 'Level', '', 'Daily', 'Amounts', 'Variations', 'Males', 
                       'Proportion', '1998', 'by', 'Models', 'A', 'Efficient', 'Lighter', 'Higher', 'Processing', 'Work', 'Additional', 
                       'Gaining', 'Improving', 'improving', 'TWI', 'Engaging', 'Fundamental', 'Aberdeen', 'For', '3/4', 'University', 
                       'Brighton', '4-year', 'CCC', 'Cambridge', 'Sparkling', '3', 'Well', 'Cardiff', 'Mix', 'Sheffield', 'Computer', 
                       'ABB', 'This', 'Governments', 'Too', 'We', 'When', 'Over', 'To', 'Tony', 'Carol', 'Ros', 'Peter', 'Janet', 
                       'Richard', 'John', 'Patricia', 'There', 'Access', 'Public', 'SMEs', 'As', 'Large', 'Their', 'Separate', 'Also', 
                       'Starting', 'Longer', 'Reforms', 'Many', 'Proportionately', 'Several', 'An', 'Outputs', 'At', 'Class', 
                       'Sixth-form', 'It', 'Most', 'All', 'Regular', 'Almost', 'Students', 'Senior', 'In', 'Teachers', 'Although', 
                       'Area-wide', 'Much', 'Colleges', 'Five', 'During', 'Basic', 'National', 'Support', 'Even', 'Youth', 'NVYOs', 
                       'Standards', 'DfES', '35.1', 'Physical', 'Improved', 'Advances', 'Commercial', '1', 'Goddard', 'Todd', 
                       'Smith', 'Gill', 'Human', 'Her', 'Further', 'Kessel', 'Source', 'Notes', 'Being', 'Ensuring', 'Identifying', 
                       'radiation', 'occupational', 'liaison', '2001/02', 'Outturn', 'Overall', 'Building', 'Defence', 'Following', 
                       'MOD', 'Emphasis', 'INDUSTRIAL', '(A)', 'INDUSTRY', 'Analysis', 'Well-established', 'Good', 'Easy', 'Lack', 
                       'Small', 'End', 'Markets', 'Decline', 'Rapid', 'Strategic', 'Focus', 'Development', 'Modelling', 'Recommendations', 
                       'limit', 'stop', 'an', 'Part', 'any']
unwanted_pos_tags = ['$', "''", '(', ')', ':','IN','LS','MD','PP','TO','VBD','VBN','WDT','WP$','WRB','``','DT','CC','WP']
POS_conversion_list = {'VBG':'V','VB':'V','VBZ':'V', 'NNS':'N', 'JJ':'J', 'NN':'N', 'NP':'N', 'VVD':'V', 
                       'VBP':'V', 'VVP':'V', 'CD':'N', 'RB':'R', 'VVN':'V', 'VVZ':'V', 'PDT':'N', 
                       'VVG':'V', 'VV':'V', 'JJS':'J', 'JJR':'J', 'RBS':'R', 'RBR':'R', 'RP':'R', 'NPS':'N', 'SYM':'N', 'FW':'N', 'UH':'N'}

texts = bs_data.find_all("text")

for i in range(len(texts)):
    texts[i] = texts[i].find_all("s")
    
for i in tqdm(range(len(texts))):
    for j in range(len(texts[i])):
        temp_text = texts[i][j].text.split("\n")[1:-1]
        texts[i][j] = []
        for k in range(len(temp_text)):
            temp_text[k] = temp_text[k].split("\t")
            """
            The part from here is to put back in 
            place some POS tags in their right place
            """
            if len(temp_text[k])!=3:
                temp_text[k] = [temp_text[k][1], temp_text[k][2], temp_text[k][4]]
                #print(temp_text[k])
            
            current_pos = temp_text[k][1]
            
            if current_pos in key_words_to_invert:
                temp_text[k][1] = temp_text[k][2]
                temp_text[k][2] = current_pos
            
            """
            This part is to remove the stop words and punctuation from the set
            """
            if (temp_text[k][2].lower() not in stop_words) and (temp_text[k][1] not in unwanted_pos_tags):
                #texts[i][j][0].append(temp_text[k][0])
                word_pos = temp_text[k][2].lower() + "." + POS_conversion_list[temp_text[k][1]]
                texts[i][j].append(word_pos)
                
new_filepath = args.corpus_xml_fp
new_filepath = new_filepath.split(".")
new_filepath = "..\\datasets\\UKWAC\\UKWAC-1.xml\\UKWAC-1_retrieval_ready.pickle"
pickle.dump(texts, open(new_filepath,"wb"))