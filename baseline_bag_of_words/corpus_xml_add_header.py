import os
from tqdm import tqdm
UKWAC_dir = "C:/Users/NICOLAS/Documents/KULeuven/master_thesis/datasets/UKWAC/"

for i in tqdm(range(4,25)):
    filepath = os.path.join(UKWAC_dir, "UKWAC-"+str(i+1)+".xml")

    f = open(filepath,'r',encoding='latin5')

    new_filepath = os.path.join(UKWAC_dir, "UKWAC-"+str(i+1)+"_processing_ready.xml")
    f_new = open(new_filepath,"w",encoding='latin5')

    f_new.write("<corpus>\n")
    buffer = f.read()
    f_new.write(buffer)
    f.close()
    f_new.write("</corpus>\n")
    f_new.close()