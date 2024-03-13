import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-fp", "--corpus-xml-fp", type=str, help="path of the corpus xml dataset. It has to be corrected so that all the xml is into on root node, which is not the case by default. This means having to pass the file to corpus_xml_modifier.py")
args = parser.parse_args()

f = open(args.corpus_xml_fp,'r',encoding='latin5')

new_filepath = args.corpus_xml_fp
new_filepath = new_filepath.split(".")
new_filepath = new_filepath[0] + "_processing_ready.xml"

f_new = open(new_filepath,"w",encoding='latin5')

f_new.write("<corpus>\n")
buffer = f.readline()
while buffer != "":
    f_new.write(buffer)
    buffer = f.readline()
f.close()
f_new.close()